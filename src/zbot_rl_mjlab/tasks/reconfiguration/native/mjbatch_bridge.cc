// CPU batch rollout using mjbatch's Apache-2.0 ThreadPool.
#include <mujoco/mujoco.h>
#include "threadpool.h"
#include <vector>
#include <memory>
struct Bridge {
  const mjModel* m; ThreadPool pool; std::vector<mjData*> workers;
  Bridge(const mjModel* model, int n):m(model),pool(n){for(int i=0;i<n;i++)workers.push_back(mj_makeData(m));}
  ~Bridge(){for(auto d:workers)mj_deleteData(d);}
};
extern "C" {
void* zbot_batch_create(const mjModel* m,int n){try{return new Bridge(m,n);}catch(...){return nullptr;}}
void zbot_batch_destroy(void* p){delete static_cast<Bridge*>(p);}
void zbot_batch_rollout(void* p,const mjData* source,int count,int steps,const double* ctrl,const int* act,const int* eq,const int* root,const int* site,const int* body,double* out){
 auto& b=*static_cast<Bridge*>(p);const mjModel* m=b.m;
 b.pool.Run(count,[&](int t,int i){
  mjData* d=b.workers[t];mj_copyData(d,m,source);
  for(int j=0;j<6;j++)d->ctrl[act[j]]=ctrl[6*i+j];
  for(int k=0;k<steps;k++)mj_step(m,d);
  for(int f=0;f<6;f++){
   double g[6]={};
   for(int r=0;r<d->nefc;r++)if(d->efc_type[r]==mjCNSTR_EQUALITY&&d->efc_id[r]==eq[f])
    for(int k=0;k<6;k++)g[k]+=d->efc_J[r*m->nv+root[f]+k]*d->efc_force[r];
   double arm[3],moment[3]={};const double* R=d->xmat+9*body[f];const double* P=d->site_xmat+9*site[f];
   for(int k=0;k<3;k++){arm[k]=d->site_xpos[3*site[f]+k]-d->xpos[3*body[f]+k];for(int j=0;j<3;j++)moment[k]+=R[3*k+j]*g[3+j];}
   double cross[3]={arm[1]*g[2]-arm[2]*g[1],arm[2]*g[0]-arm[0]*g[2],arm[0]*g[1]-arm[1]*g[0]};
   for(int k=0;k<3;k++){double force=0,torque=0;for(int j=0;j<3;j++){force+=P[3*j+k]*g[j];torque+=P[3*j+k]*(moment[j]-cross[j]);}out[36*i+6*f+k]=force;out[36*i+6*f+3+k]=torque;}
  }
 });
}
}
