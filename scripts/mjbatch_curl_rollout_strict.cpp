// Multi-step CPU rollouts using mjbatch's upstream persistent ThreadPool.
// Each worker owns one MjData; every candidate is restored from the same state.
#include <mujoco/mujoco.h>
#include "threadpool.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

struct Face { int oldeq, neweq, root, site, body; };
struct GeomPair { int a,b,face; };
struct Batch {
  mjModel* m; ThreadPool pool; std::vector<mjData*> data;
  std::array<Face, 6> face; std::array<int, 6> actuator, dof, qpos;
  std::vector<int> module; std::vector<GeomPair> pairs;
  Batch(mjModel* model, int threads): m(model), pool(threads), module(m->ngeom,-1) {
    for (int t=0;t<threads;t++) data.push_back(mj_makeData(m));
    for (int i=0;i<6;i++) {
      std::string prefix="module_"+std::to_string(i)+"/";
      int root=mj_name2id(m,mjOBJ_JOINT,(prefix+"module_root").c_str());
      int joint=mj_name2id(m,mjOBJ_JOINT,(prefix+"joint"+std::to_string(i+1)).c_str());
      face[i]={mj_name2id(m,mjOBJ_EQUALITY,("connector_"+std::to_string(i)).c_str()),
        i<5?mj_name2id(m,mjOBJ_EQUALITY,("connector_"+std::to_string(i)+"_rz270").c_str()):-1,
        m->jnt_dofadr[root],mj_name2id(m,mjOBJ_SITE,(prefix+"port_out").c_str()),m->jnt_bodyid[root]};
      actuator[i]=mj_name2id(m,mjOBJ_ACTUATOR,(prefix+"joint"+std::to_string(i+1)).c_str());
      dof[i]=m->jnt_dofadr[joint];qpos[i]=m->jnt_qposadr[joint];
    }
    for(int g=0;g<m->ngeom;g++) {
      const char* n=mj_id2name(m,mjOBJ_BODY,m->geom_bodyid[g]);
      if(n && std::string(n).rfind("module_",0)==0) module[g]=std::atoi(n+7);
    }
    std::array<int,6> mateout,matein;
    for(int face=0;face<6;face++){
      int other=(face+1)%6;
      mateout[face]=mj_name2id(m,mjOBJ_GEOM,("module_"+std::to_string(face)+"/b"+std::to_string(face+1)+"_geom").c_str());
      matein[face]=mj_name2id(m,mjOBJ_GEOM,("module_"+std::to_string(other)+"/a"+std::to_string(other+1)+"_geom").c_str());
    }
    for(int a=0;a<m->ngeom;a++)for(int b=a+1;b<m->ngeom;b++){
      if(module[a]<0||module[b]<0||module[a]==module[b])continue;
      int face=-1;
      for(int i=0;i<6;i++)if((a==mateout[i]&&b==matein[i])||(a==matein[i]&&b==mateout[i])){face=i;break;}
      pairs.push_back({a,b,face});
    }
  }
  ~Batch(){for(auto d:data)mj_deleteData(d);mj_deleteModel(m);}
};
extern "C" void* zbatch_create(const char* filename,int threads) {
  mjModel* m=mj_loadModel(filename,nullptr);if(!m||threads<1) return nullptr;
  if(mj_isSparse(m)) {mj_deleteModel(m);return nullptr;}
  return new Batch(m,threads);
}
extern "C" void zbatch_free(void* ptr){delete static_cast<Batch*>(ptr);}
// result fields: status, max load, max torque, max module depth, max ground depth,
// failure step, failure face (1-based), plus six individual maximum loads.
static int zbatch_eval_impl(void* ptr,const double* state,int count,int maxsteps,
    const int* lengths,const double* controls,double* result,double* endstate,double load_limit,double* metrics,int state_stride,double* minimum_distance) {
  auto& b=*static_cast<Batch*>(ptr);const mjModel* m=b.m;
  int ns=mj_stateSize(m,mjSTATE_INTEGRATION);
  b.pool.Run(count,[&](int worker,int candidate){
    const double* candidate_state=state+static_cast<size_t>(candidate)*state_stride;
    mjData* d=b.data[worker];mj_setState(m,d,candidate_state,mjSTATE_INTEGRATION);
    mj_forward(m,d);
    double initial_R[9];mju_copy(initial_R,d->xmat+9*b.face[0].body,9);
    double axis[3];for(int j=0;j<3;j++)axis[j]=d->xpos[3*b.face[5].body+j]-d->xpos[3*b.face[0].body+j];
    mju_normalize3(axis);double* stats=metrics+candidate*4;std::fill(stats,stats+4,0.);stats[3]=1e9;
    mj_setState(m,d,candidate_state,mjSTATE_INTEGRATION);
    double* out=result+candidate*13;std::fill(out,out+13,0.);out[5]=-1;out[6]=-1;
    double closest=.05;
    int overload[6]={};
    int warnings[mjNWARNING];for(int i=0;i<mjNWARNING;i++)warnings[i]=d->warning[i].number;
    for(int step=0;step<lengths[candidate];step++) {
      const double* ctrl=controls+(static_cast<size_t>(candidate)*maxsteps+step)*6;
      bool fail=false;
      for(int j=0;j<6;j++) {
        double kp=m->actuator_gainprm[b.actuator[j]*mjNGAIN];
        double kd=-m->actuator_biasprm[b.actuator[j]*mjNBIAS+2];
        double demand=kp*(ctrl[j]-d->qpos[b.qpos[j]])-kd*d->qvel[b.dof[j]];
        // MuJoCo's configured actuator force saturation enforces ±20 Nm;
        // reject nonfinite commands and audit the actual force after stepping.
        if(!std::isfinite(demand)) {out[0]=2;fail=true;break;}
        d->ctrl[b.actuator[j]]=ctrl[j];
      }
      if(fail){out[5]=step;break;}
      mj_step(m,d);
      // Velocity, source-asset hinge limits and warnings after the step.
      for (int j=0;j<6;j++) {
        if (std::abs(d->qvel[b.dof[j]]) > 2*mjPI) {out[0]=7; fail=true; break;}
        // ZBOT6.xml module_joint range, checked by the Python caller.
        if (std::abs(d->qpos[b.qpos[j]]) > 6.28) {out[0]=8; fail=true; break;}
      }
      for(int i=0;i<mjNWARNING;i++)if(d->warning[i].number>warnings[i]){out[0]=9;fail=true;break;}
      if(fail){out[5]=step;break;}
      for(int i=0;i<6;i++){
        const double* omega=d->cvel+6*b.face[i].body;
        stats[0]=std::max(stats[0],mju_norm3(omega));
      }
      double rr[9],quat[4];mju_mulMatMatT(rr,d->xmat+9*b.face[0].body,initial_R,3,3,3);
      mju_mat2Quat(quat,rr);
      double roll=2*std::atan2(quat[1]*axis[0]+quat[2]*axis[1]+quat[3]*axis[2],quat[0]);
      roll=std::remainder(roll,2*mjPI);stats[2]=std::max(stats[2],std::abs(roll));
      double mass=0.,z=0.;for(int i=1;i<m->nbody;i++){mass+=m->body_mass[i];z+=m->body_mass[i]*d->xipos[3*i+2];}
      if(mass>0)stats[1]=std::max(stats[1],z/mass);
      int grounded=0;for(int c=0;c<d->ncon;c++)if(!d->contact[c].exclude &&
          ((b.module[d->contact[c].geom1]>=0)!=(b.module[d->contact[c].geom2]>=0)))grounded++;
      stats[3]=std::min(stats[3],double(grounded));
      for(int i=0;i<6;i++) {
        const Face a=b.face[i];int eq=(a.neweq>=0&&d->eq_active[a.neweq])?a.neweq:a.oldeq;
        if(!d->eq_active[eq]) continue;
        double g[6]={};int rows=0;
        for(int r=0;r<d->nefc;r++) if(d->efc_type[r]==mjCNSTR_EQUALITY&&d->efc_id[r]==eq) {
          rows++;for(int j=0;j<6;j++)g[j]+=d->efc_J[r*m->nv+a.root+j]*d->efc_force[r];
        }
        if(rows!=6){out[0]=5;fail=true;break;}
        const double* R=d->xmat+9*a.body;const double* P=d->site_xmat+9*a.site;
        double arm[3],moment[3]={},force[3]={},localm[3]={};
        for(int k=0;k<3;k++) {arm[k]=d->site_xpos[3*a.site+k]-d->xpos[3*a.body+k];for(int j=0;j<3;j++)moment[k]+=R[3*k+j]*g[3+j];}
        double cross[3]={arm[1]*g[2]-arm[2]*g[1],arm[2]*g[0]-arm[0]*g[2],arm[0]*g[1]-arm[1]*g[0]};
        for(int k=0;k<3;k++)for(int j=0;j<3;j++){force[k]+=P[3*j+k]*g[j];localm[k]+=P[3*j+k]*(moment[j]-cross[j]);}
        double load=std::max(force[2],0.)+std::hypot(localm[0],localm[1])/.05;
        out[7+i]=std::max(out[7+i],load);out[1]=std::max(out[1],load);
        if(!std::isfinite(load)){out[0]=5;fail=true;break;}
        if(load>load_limit) {out[0]=1;out[6]=i+1;fail=true;break;}
      }
      if(fail){out[5]=step;break;}
      for(int j=0;j<6;j++) {
        double tau=std::abs(d->qfrc_actuator[b.dof[j]]);out[2]=std::max(out[2],tau);
        overload[j]=tau>14.?overload[j]+1:0;
        if(tau>20.+1e-6||overload[j]>=1000){out[0]=2;fail=true;break;}
      }
      if(fail){out[5]=step;break;}
      for(int c=0;c<d->ncon;c++) {
        const mjContact& contact=d->contact[c];if(contact.exclude)continue;
        int a=b.module[contact.geom1],bb=b.module[contact.geom2];double depth=std::max(0.,-double(contact.dist));
        if(a>=0&&bb>=0&&a!=bb)out[3]=std::max(out[3],depth);
        if((a>=0)!=(bb>=0))out[4]=std::max(out[4],depth);
      }
      if(out[3]>.002||out[4]>.003){out[0]=3;out[5]=step;break;}
      // Wrenches/contact were read above from the solver's pre-integration
      // state. Refresh geometry only after those measurements, so the mesh
      // audit uses the integrated qpos without mixing wrench timestamps.
      mj_kinematics(m,d);
      for(const auto& pair:b.pairs){
        if(fail)break;
        int ga=pair.a,gb=pair.b;bool mating=false;int face=pair.face;
        if(face>=0){
          int eq=b.face[face].oldeq;
          if(b.face[face].neweq>=0 && d->eq_active[b.face[face].neweq]) eq=b.face[face].neweq;
          if(d->eq_active[eq]) mating=true;
          else {
            int s=m->eq_obj1id[eq],t=m->eq_obj2id[eq];
            double dx[3];for(int k=0;k<3;k++)dx[k]=d->site_xpos[3*s+k]-d->site_xpos[3*t+k];
            const double* A=d->site_xmat+9*s;const double* B=d->site_xmat+9*t;
            double trace=0;for(int k=0;k<9;k++)trace+=A[k]*B[k];
            double angle=std::acos(std::clamp((trace-1)*.5,-1.,1.));
            if(mju_norm3(dx)<.001 && angle<2*mjPI/180.) mating=true;
            if(!mating && b.face[face].neweq>=0){
              eq=b.face[face].neweq;
              s=m->eq_obj1id[eq];t=m->eq_obj2id[eq];
              for(int k=0;k<3;k++)dx[k]=d->site_xpos[3*s+k]-d->site_xpos[3*t+k];
              A=d->site_xmat+9*s;B=d->site_xmat+9*t;trace=0;
              for(int k=0;k<9;k++)trace+=A[k]*B[k];
              angle=std::acos(std::clamp((trace-1)*.5,-1.,1.));
              if(mju_norm3(dx)<.001 && angle<2*mjPI/180.) mating=true;
            }
          }
        }
        if(mating)continue;
        double dx[3];for(int k=0;k<3;k++)dx[k]=d->geom_xpos[3*ga+k]-d->geom_xpos[3*gb+k];
        double cutoff=minimum_distance ? .05 : .001;
        if(mju_norm3(dx)>m->geom_rbound[ga]+m->geom_rbound[gb]+cutoff)continue;
        double distance=mj_geomDistance(m,d,ga,gb,cutoff,nullptr);
        closest=std::min(closest,distance);
        if(distance<=0.){out[0]=6;fail=true;break;}
      }
      if(fail){out[5]=step;break;}
      for(int j=0;j<m->nq;j++)if(!std::isfinite(d->qpos[j]))fail=true;
      for(int j=0;j<m->nv;j++)if(!std::isfinite(d->qvel[j]))fail=true;
      if(fail){out[0]=4;out[5]=step;break;}
    }
    if(minimum_distance) minimum_distance[candidate]=closest;
    mj_getState(m,d,endstate+static_cast<size_t>(candidate)*ns,mjSTATE_INTEGRATION);
  });
  return 0;
}

extern "C" int zbatch_eval(void* ptr,const double* state,int count,int maxsteps,
    const int* lengths,const double* controls,double* result,double* endstate,double load_limit,double* metrics,int state_stride) {
  return zbatch_eval_impl(ptr,state,count,maxsteps,lengths,controls,result,endstate,load_limit,metrics,state_stride,nullptr);
}
extern "C" int zbatch_eval_distance(void* ptr,const double* state,int count,int maxsteps,
    const int* lengths,const double* controls,double* result,double* endstate,double load_limit,double* metrics,int state_stride,double* minimum_distance) {
  return zbatch_eval_impl(ptr,state,count,maxsteps,lengths,controls,result,endstate,load_limit,metrics,state_stride,minimum_distance);
}
