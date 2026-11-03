// Research-only MuJoCo SDF for a fixed 100 mm disk with four 4.2 mm holes.
// No hardware calibration; `max` CSG is not an exact SDF near intersections.
// Contact depth and mj_geomDistance are known to be unreliable for this model.
// Do not use this plugin for control, collision safety, or production scenes.
#include <mujoco/mujoco.h>
#include <cmath>
#include <algorithm>
static double sdf(const double* p) {
 double q1=std::hypot(p[0],p[1])-.05, q2=std::abs(p[2]+.001)-.001;
 double d=std::min(std::max(q1,q2),0.)+std::hypot(std::max(q1,0.),std::max(q2,0.));
 double x[4]={.035,0,-.035,0}, y[4]={0,.035,0,-.035};
 for(int i=0;i<4;i++) d=std::max(d, .0021-std::hypot(p[0]-x[i],p[1]-y[i]));
 return d;
}
static void grad(double* g,const double* p) {
 const double rr=std::hypot(p[0],p[1]); double q1=rr-.05, q2=std::abs(p[2]+.001)-.001;
 double positive1=std::max(q1,0.),positive2=std::max(q2,0.);
 double len=std::hypot(positive1,positive2); double ar=0.,az=0.;
 if(len>0) { ar=positive1/len;az=positive2/len; }
 else if(q1>q2) ar=1.; else az=1.;
 g[0]=rr>1e-15?ar*p[0]/rr:0.;g[1]=rr>1e-15?ar*p[1]/rr:0.;g[2]=az*(p[2]+.001>=0?1.:-1.);
 double d=std::min(std::max(q1,q2),0.)+len;
 const double x[4]={.035,0,-.035,0}, y[4]={0,.035,0,-.035};
 for(int i=0;i<4;i++) {
   double dx=p[0]-x[i],dy=p[1]-y[i], r=std::hypot(dx,dy), hole=.0021-r;
   if(hole>d) {d=hole;g[0]=r>1e-15?-dx/r:0.;g[1]=r>1e-15?-dy/r:0.;g[2]=0.;}
 }
}
extern "C" int register_zbot_sdf() {
 mjpPlugin p; mjp_defaultPlugin(&p);p.name="zbot.sdf.experiment";p.capabilityflags=mjPLUGIN_SDF;
 p.nstate=[](const mjModel*,int){return 0;};p.init=[](const mjModel*,mjData*,int){return 0;};
 p.reset=[](const mjModel*,mjtNum*,void*,int){};p.compute=[](const mjModel*,mjData*,int,int){};
 p.sdf_distance=[](const mjtNum* x,const mjData*,int){return sdf(x);};
 p.sdf_gradient=[](mjtNum* g,const mjtNum* x,const mjData*,int){grad(g,x);};
 p.sdf_staticdistance=[](const mjtNum* x,const mjtNum*){return sdf(x);};
 p.sdf_aabb=[](mjtNum* a,const mjtNum*){a[0]=0;a[1]=0;a[2]=-.001;a[3]=.05;a[4]=.05;a[5]=.001;};
 p.sdf_attribute=[](mjtNum*,const char**,const char**){};
 return mjp_registerPlugin(&p);
}
