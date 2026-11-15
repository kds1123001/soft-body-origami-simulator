#include <math.h>

typedef struct { double x, y, z; } Vec3;
static Vec3 sub(Vec3 a, Vec3 b) { return (Vec3){a.x-b.x,a.y-b.y,a.z-b.z}; }
static double dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
static Vec3 cross(Vec3 a, Vec3 b) {
    return (Vec3){a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
static double norm(Vec3 a) { return sqrt(dot(a,a)+1e-24); }

double so_signed_dihedral(const double *x, int i, int j, int a, int b) {
    Vec3 p0={x[3*i],x[3*i+1],x[3*i+2]};
    Vec3 p1={x[3*j],x[3*j+1],x[3*j+2]};
    Vec3 pa={x[3*a],x[3*a+1],x[3*a+2]};
    Vec3 pb={x[3*b],x[3*b+1],x[3*b+2]};
    Vec3 e=sub(p1,p0), n0=cross(e,sub(pa,p0)), n1=cross(sub(pb,p0),e);
    double el=norm(e), n0l=norm(n0), n1l=norm(n1);
    n0=(Vec3){n0.x/n0l,n0.y/n0l,n0.z/n0l};
    n1=(Vec3){n1.x/n1l,n1.y/n1l,n1.z/n1l};
    Vec3 t={e.x/el,e.y/el,e.z/el};
    return atan2(dot(cross(n0,n1),t),dot(n0,n1));
}

double so_edge_strain(const double *x, int i, int j, double rest) {
    Vec3 a={x[3*i],x[3*i+1],x[3*i+2]};
    Vec3 b={x[3*j],x[3*j+1],x[3*j+2]};
    return norm(sub(a,b))-rest;
}
