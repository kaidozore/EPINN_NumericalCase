#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void steel02_kernel(
    const double* u, int batch, int steps, int reduced,
    const double* curvature, const double* force_map, int elements, int gauss,
    const double* fiber_y, const double* fiber_area, int fibers,
    const double* p, double* state, double* internal, double* tangent) {
  const int point=blockIdx.x*blockDim.x+threadIdx.x;
  const int total=batch*elements*gauss;
  if (point>=total) return;
  const int ig=point%gauss;
  const int ie=(point/gauss)%elements;
  const int ib=point/(gauss*elements);
  const int states=11;
  const double* B=curvature+(ie*gauss+ig)*reduced;
  const double* P=force_map+(ie*gauss+ig)*reduced;
  const double E0=p[0], Fy=p[1], b=p[2], R0=p[3], cR1=p[4], cR2=p[5];
  const double a1=p[6], a2=p[7], a3=p[8], a4=p[9];
  const double epsy=Fy/E0, esh=b*E0;
  for (int it=0; it<steps; ++it) {
    double kappa=0.0;
    for (int ir=0; ir<reduced; ++ir)
      kappa += u[(static_cast<long long>(ib)*steps+it)*reduced+ir]*B[ir];
    double moment=0.0, section_tangent=0.0;
    for (int jf=0; jf<fibers; ++jf) {
      double* s=state+(((static_cast<long long>(ib)*elements+ie)*gauss+ig)*fibers+jf)*states;
      const double strain=-kappa*fiber_y[jf], deps=strain-s[0];
      double stress=s[1], et=s[2];
      if (fabs(deps)>1.0e-14) {
        double epspl=s[4], epss0=s[5], sigs0=s[6], epsr=s[7], sigr=s[8];
        double epsmax=s[9], epsmin=s[10]; int kon=static_cast<int>(s[3]);
        if (kon==0 || kon==3) {
          epsmax=epsy; epsmin=-epsy;
          if (deps<0) {kon=2; epss0=-epsy; sigs0=-Fy; epspl=-epsy;}
          else {kon=1; epss0=epsy; sigs0=Fy; epspl=epsy;}
        }
        if (kon==2 && deps>0) {
          const double oldmax=epsmax; epsmin=fmin(epsmin,s[0]);
          const double shift=1.0+a3*pow((oldmax-epsmin)/(2.0*a4*epsy),0.8);
          epss0=(Fy*shift-esh*epsy*shift-s[1]+E0*s[0])/(E0-esh);
          sigs0=Fy*shift+esh*(epss0-epsy*shift);
          kon=1; epsr=s[0]; sigr=s[1]; epspl=oldmax;
        }
        if (kon==1 && deps<0) {
          const double oldmin=epsmin; epsmax=fmax(epsmax,s[0]);
          const double shift=1.0+a1*pow((epsmax-oldmin)/(2.0*a2*epsy),0.8);
          epss0=(-Fy*shift+esh*epsy*shift-s[1]+E0*s[0])/(E0-esh);
          sigs0=-Fy*shift+esh*(epss0+epsy*shift);
          kon=2; epsr=s[0]; sigr=s[1]; epspl=oldmin;
        }
        const double xi=fabs((epspl-epss0)/epsy);
        const double radius=R0*(1.0-cR1*xi/(cR2+xi));
        double denominator=epss0-epsr;
        if (fabs(denominator)<=1.0e-18) denominator=1.0;
        const double ratio=(strain-epsr)/denominator;
        const double dum1=1.0+pow(fabs(ratio),radius), dum2=pow(dum1,1.0/radius);
        stress=(b*ratio+(1.0-b)*ratio/dum2)*(sigs0-sigr)+sigr;
        et=(b+(1.0-b)/(dum1*dum2))*(sigs0-sigr)/denominator;
        s[0]=strain; s[1]=stress; s[2]=et; s[3]=static_cast<double>(kon);
        s[4]=epspl; s[5]=epss0; s[6]=sigs0; s[7]=epsr; s[8]=sigr;
        s[9]=epsmax; s[10]=epsmin;
      }
      moment -= stress*fiber_area[jf]*fiber_y[jf];
      section_tangent += et*fiber_area[jf]*fiber_y[jf]*fiber_y[jf];
    }
    double* f=internal+(static_cast<long long>(ib)*steps+it)*reduced;
    double* kt=tangent+(static_cast<long long>(ib)*steps+it)*reduced*reduced;
    for (int ir=0; ir<reduced; ++ir) {
      atomicAdd(f+ir,moment*P[ir]);
      for (int jr=0; jr<reduced; ++jr)
        atomicAdd(kt+ir*reduced+jr,section_tangent*P[ir]*B[jr]);
    }
  }
}

std::vector<torch::Tensor> steel02_cuda_forward(
    torch::Tensor displacement, torch::Tensor curvature,
    torch::Tensor force_map, torch::Tensor fiber_y,
    torch::Tensor fiber_area, torch::Tensor parameters, torch::Tensor state) {
  TORCH_CHECK(displacement.is_cuda() && displacement.scalar_type()==torch::kFloat64,
              "Steel02 CUDA requires CUDA float64 tensors");
  auto u=displacement.contiguous(); auto B=curvature.contiguous();
  auto P=force_map.contiguous(); auto y=fiber_y.contiguous();
  auto area=fiber_area.contiguous(); auto p=parameters.contiguous();
  auto final_state=state.contiguous().clone();
  const int batch=u.size(0), steps=u.size(1), reduced=u.size(2);
  const int elements=B.size(0), gauss=B.size(1), fibers=y.numel();
  auto internal=torch::zeros({batch,steps,reduced},u.options());
  auto tangent=torch::zeros({batch,steps,reduced,reduced},u.options());
  const int total=batch*elements*gauss, threads=128;
  steel02_kernel<<<(total+threads-1)/threads,threads,0,c10::cuda::getCurrentCUDAStream().stream()>>>(
      u.data_ptr<double>(),batch,steps,reduced,B.data_ptr<double>(),P.data_ptr<double>(),
      elements,gauss,y.data_ptr<double>(),area.data_ptr<double>(),fibers,p.data_ptr<double>(),
      final_state.data_ptr<double>(),internal.data_ptr<double>(),tangent.data_ptr<double>());
  auto error=cudaGetLastError();
  TORCH_CHECK(error==cudaSuccess,"Steel02 CUDA kernel failed: ",cudaGetErrorString(error));
  return {internal,tangent,final_state};
}
