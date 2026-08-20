#include "dsp_features.h"
#include <math.h>
#include <string.h>

#define IIS3DWB_MS2_PER_LSB (0.000488f * 9.80665f)

bool rg_compute_vibration_features(const rg_xyz_i16_t *s,size_t n,float fs,rg_vibration_features_t *o){
    if(!s||!o||n<8||fs<=0.0f) return false;
    memset(o,0,sizeof *o);
    double mean[3]={0,0,0};for(size_t i=0;i<n;i++){mean[0]+=s[i].x;mean[1]+=s[i].y;mean[2]+=s[i].z;}for(unsigned a=0;a<3;a++)mean[a]/=(double)n;
    double axis2[3]={0,0,0},sum2=0,sum4=0,peak=0,mag_mean=0;
    // First pass computes de-meaned vector magnitude and moments. Removing per-axis DC
    // prevents gravity/mounting bias from dominating a vibration window.
    for(size_t i=0;i<n;i++){
        const double ax=((double)s[i].x-mean[0])*IIS3DWB_MS2_PER_LSB;
        const double ay=((double)s[i].y-mean[1])*IIS3DWB_MS2_PER_LSB;
        const double az=((double)s[i].z-mean[2])*IIS3DWB_MS2_PER_LSB;
        axis2[0]+=ax*ax;axis2[1]+=ay*ay;axis2[2]+=az*az;
        const double m=sqrt(ax*ax+ay*ay+az*az);mag_mean+=m;sum2+=m*m;sum4+=m*m*m*m;if(m>peak)peak=m;
    }
    mag_mean/=(double)n;for(unsigned a=0;a<3;a++)o->axis_rms[a]=(float)sqrt(axis2[a]/n);
    const double m2=sum2/n;o->rms=(float)sqrt(m2);o->peak=(float)peak;o->kurtosis=m2>1e-18?(float)((sum4/n)/(m2*m2)):0.0f;o->crest_factor=o->rms>1e-9f?o->peak/o->rms:0.0f;
    const double pi=3.14159265358979323846;const float freqs[4]={25.f,75.f,200.f,500.f};
    for(unsigned b=0;b<4;b++){
        const double w=2.0*pi*freqs[b]/fs,c=2.0*cos(w);double q0=0,q1=0,q2=0;
        for(size_t i=0;i<n;i++){
            const double ax=((double)s[i].x-mean[0])*IIS3DWB_MS2_PER_LSB,ay=((double)s[i].y-mean[1])*IIS3DWB_MS2_PER_LSB,az=((double)s[i].z-mean[2])*IIS3DWB_MS2_PER_LSB;
            const double m=sqrt(ax*ax+ay*ay+az*az)-mag_mean;q0=m+c*q1-q2;q2=q1;q1=q0;
        }
        const double power=q1*q1+q2*q2-c*q1*q2;o->band_energy[b]=(float)(power/n);
    }
    return true;
}
