#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

extern "C" EXPORT void steel02_forward(
    const double* u, int batch, int steps, int reduced,
    const double* curvature, const double* force_map,
    int elements, int gauss, const double* fiber_y,
    const double* fiber_area, int fibers, const double* p,
    const double* initial_state, double* internal,
    double* tangent, double* final_state) {
  const int states = 11;
  const std::size_t state_count =
      static_cast<std::size_t>(batch) * elements * gauss * fibers * states;
  std::memcpy(final_state, initial_state, state_count * sizeof(double));

  const double E0=p[0], Fy=p[1], b=p[2], R0=p[3], cR1=p[4], cR2=p[5];
  const double a1=p[6], a2=p[7], a3=p[8], a4=p[9];
  const double epsy=Fy/E0, esh=b*E0;

  #pragma omp parallel for
  for (int ib=0; ib<batch; ++ib) {
    for (int it=0; it<steps; ++it) {
      double* f = internal + (static_cast<std::size_t>(ib)*steps+it)*reduced;
      double* kt = tangent + (static_cast<std::size_t>(ib)*steps+it)*reduced*reduced;
      for (int ie=0; ie<elements; ++ie) for (int ig=0; ig<gauss; ++ig) {
        const double* B = curvature + (ie*gauss+ig)*reduced;
        const double* P = force_map + (ie*gauss+ig)*reduced;
        double kappa=0.0;
        for (int ir=0; ir<reduced; ++ir)
          kappa += u[(static_cast<std::size_t>(ib)*steps+it)*reduced+ir]*B[ir];
        double moment=0.0, section_tangent=0.0;
        for (int jf=0; jf<fibers; ++jf) {
          double* s = final_state + (((static_cast<std::size_t>(ib)*elements+ie)*gauss+ig)*fibers+jf)*states;
          const double strain=-kappa*fiber_y[jf];
          const double deps=strain-s[0];
          double stress=s[1], et=s[2];
          if (std::abs(deps)>1.0e-14) {
            double epspl=s[4], epss0=s[5], sigs0=s[6], epsr=s[7], sigr=s[8];
            double epsmax=s[9], epsmin=s[10]; int kon=static_cast<int>(s[3]);
            if (kon==0 || kon==3) {
              epsmax=epsy; epsmin=-epsy;
              if (deps<0) {kon=2; epss0=-epsy; sigs0=-Fy; epspl=-epsy;}
              else {kon=1; epss0=epsy; sigs0=Fy; epspl=epsy;}
            }
            if (kon==2 && deps>0) {
              const double oldmax=epsmax; epsmin=std::min(epsmin,s[0]);
              const double d1=(oldmax-epsmin)/(2.0*a4*epsy);
              const double shift=1.0+a3*std::pow(d1,0.8);
              epss0=(Fy*shift-esh*epsy*shift-s[1]+E0*s[0])/(E0-esh);
              sigs0=Fy*shift+esh*(epss0-epsy*shift);
              kon=1; epsr=s[0]; sigr=s[1]; epspl=oldmax;
            }
            if (kon==1 && deps<0) {
              const double oldmin=epsmin; epsmax=std::max(epsmax,s[0]);
              const double d1=(epsmax-oldmin)/(2.0*a2*epsy);
              const double shift=1.0+a1*std::pow(d1,0.8);
              epss0=(-Fy*shift+esh*epsy*shift-s[1]+E0*s[0])/(E0-esh);
              sigs0=-Fy*shift+esh*(epss0+epsy*shift);
              kon=2; epsr=s[0]; sigr=s[1]; epspl=oldmin;
            }
            const double xi=std::abs((epspl-epss0)/epsy);
            const double radius=R0*(1.0-cR1*xi/(cR2+xi));
            double denominator=epss0-epsr;
            if (std::abs(denominator)<=1.0e-18) denominator=1.0;
            const double ratio=(strain-epsr)/denominator;
            const double dum1=1.0+std::pow(std::abs(ratio),radius);
            const double dum2=std::pow(dum1,1.0/radius);
            stress=(b*ratio+(1.0-b)*ratio/dum2)*(sigs0-sigr)+sigr;
            et=(b+(1.0-b)/(dum1*dum2))*(sigs0-sigr)/denominator;
            s[0]=strain; s[1]=stress; s[2]=et; s[3]=static_cast<double>(kon);
            s[4]=epspl; s[5]=epss0; s[6]=sigs0; s[7]=epsr; s[8]=sigr;
            s[9]=epsmax; s[10]=epsmin;
          }
          moment -= stress*fiber_area[jf]*fiber_y[jf];
          section_tangent += et*fiber_area[jf]*fiber_y[jf]*fiber_y[jf];
        }
        for (int ir=0; ir<reduced; ++ir) {
          f[ir] += moment*P[ir];
          for (int jr=0; jr<reduced; ++jr)
            kt[ir*reduced+jr] += section_tangent*P[ir]*B[jr];
        }
      }
    }
  }
}
