// Exact weld-load and contact audit for states produced by mjbatch.Batch.
// Uses the upstream mjbatch Apache-2.0 thread pool, one MjData per worker.
#include <mujoco/mujoco.h>
#include "threadpool.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

struct Monitor {
  mjModel* model; ThreadPool pool; std::vector<mjData*> data;
  std::array<int,6> eq,root,body,site,dof;
  Monitor(mjModel* m,int threads):model(m),pool(threads) {
    for(int i=0;i<threads;i++)data.push_back(mj_makeData(m));
    for(int i=0;i<6;i++) {
      std::string p="module_"+std::to_string(i)+"/";
      int r=mj_name2id(m,mjOBJ_JOINT,(p+"module_root").c_str());
      int j=mj_name2id(m,mjOBJ_JOINT,(p+"joint"+std::to_string(i+1)).c_str());
      eq[i]=mj_name2id(m,mjOBJ_EQUALITY,("connector_"+std::to_string(i)).c_str());
      root[i]=m->jnt_dofadr[r];body[i]=m->jnt_bodyid[r];dof[i]=m->jnt_dofadr[j];
      site[i]=mj_name2id(m,mjOBJ_SITE,(p+"port_out").c_str());
    }
  }
  ~Monitor(){for(auto d:data)mj_deleteData(d);mj_deleteModel(model);}
};
extern "C" void* zm_create(const char* path,int threads) {
  mjModel* m=mj_loadModel(path,nullptr);if(!m||threads<1)return nullptr;
  return new Monitor(m,threads);
}
extern "C" void zm_free(void* p){delete static_cast<Monitor*>(p);}
// Each result: six measured peel loads, max actuator torque, reserved fields,
// and invalid flag. The model and integration-state layout match Batch.
extern "C" void zm_measure(void* p,const double* states,int count,double* out) {
  auto& b=*static_cast<Monitor*>(p);const mjModel* m=b.model;
  const int ns=mj_stateSize(m,mjSTATE_INTEGRATION);
  b.pool.Run(count,[&](int worker,int i){
    mjData* d=b.data[worker];double* r=out+10*i;std::fill(r,r+10,0.);
    mj_setState(m,d,states+static_cast<size_t>(i)*ns,mjSTATE_INTEGRATION);mj_forward(m,d);
    for(int j=0;j<m->nq;j++)if(!std::isfinite(d->qpos[j]))r[9]=1.;
    for(int j=0;j<m->nv;j++)if(!std::isfinite(d->qvel[j]))r[9]=1.;
    for(int f=0;f<6;f++) {
      r[6]=std::max(r[6],std::abs(d->qfrc_actuator[b.dof[f]]));
      if(!d->eq_active[b.eq[f]])continue;
      double g[6]={};int rows=0;
      for(int k=0;k<d->nefc;k++)if(d->efc_type[k]==mjCNSTR_EQUALITY&&d->efc_id[k]==b.eq[f]) {
        rows++;for(int j=0;j<6;j++)g[j]+=d->efc_J[k*m->nv+b.root[f]+j]*d->efc_force[k];
      }
      if(rows!=6){r[9]=1.;continue;}
      const double* R=d->xmat+9*b.body[f];const double* P=d->site_xmat+9*b.site[f];
      double arm[3],moment[3]={},force[3]={},localm[3]={};
      for(int k=0;k<3;k++){arm[k]=d->site_xpos[3*b.site[f]+k]-d->xpos[3*b.body[f]+k];
        for(int j=0;j<3;j++)moment[k]+=R[3*k+j]*g[3+j];}
      double cross[3]={arm[1]*g[2]-arm[2]*g[1],arm[2]*g[0]-arm[0]*g[2],arm[0]*g[1]-arm[1]*g[0]};
      for(int k=0;k<3;k++)for(int j=0;j<3;j++){force[k]+=P[3*j+k]*g[j];localm[k]+=P[3*j+k]*(moment[j]-cross[j]);}
      r[f]=std::max(force[2],0.)+std::hypot(localm[0],localm[1])/.05;
      if(!std::isfinite(r[f]))r[9]=1.;
    }
  });
}
