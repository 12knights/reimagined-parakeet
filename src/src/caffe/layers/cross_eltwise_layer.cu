#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/cross_eltwise_layer.hpp"
#include "caffe/util/math_functions.hpp"
template <typename Dtype>
__global__ void LabelDiffForward(const int n,const int num, const Dtype* A_in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    	out[index] = A_in[index/num] != A_in[index%num];
  }
}
namespace caffe {
template <typename Dtype>
void CrossEltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();  //shape:(n*n)*c*w*h
	
  caffe_gpu_set(count, Dtype(0.), top_data);
	int dim =bottom[0]->count()/bottom[0]->num();
	
  for(int k=0; k< bottom[0]->num(); k++)
	{
		for(int i=0; i<bottom[0]->num(); i++){
			caffe_copy(dim, bottom[0]->gpu_data()+k*dim, top_data+k*bottom[0]->count()+i*dim);
		}
		//caffe_copy(bottom[0]->count(),bottom[0]->gpu_data(), top_data+k*bottom[0]->count() );
		caffe_gpu_axpby(bottom[0]->count(), Dtype(-1), bottom[0]->gpu_data(), Dtype(1), top_data+k*bottom[0]->count());
	}
   	Dtype* top_label=top[1]->mutable_gpu_data();
	LabelDiffForward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->num()), CAFFE_CUDA_NUM_THREADS>>>(
      		bottom[0]->num()*bottom[0]->num(),bottom[0]->num(), bottom[1]->gpu_data(), top_label);
    	CUDA_POST_KERNEL_CHECK;
/*
	//check row major
	Dtype l1(0.0), l2(0.0);
	for(int i=0; i<dim; i++)
	{
		l1 += bottom[0]->data_at(0,i,0,0);
		
	}
	caffe_gpu_dot(dim, bottom[0]->gpu_data(), ones_.gpu_data(), &l2);
	if(std::abs(l1-l2)>1e-7)
	{
		LOG(FATAL)<<"not row major";
	}
*/
  /*
	//test code;
	for(int i=0; i<bottom[0]->num(); i++){
	  CHECK_EQ(bottom[0]->data_at(0,0,0,0)-bottom[0]->data_at(i,0,0,0), top[0]->data_at(i,0,0,0));
   CHECK_EQ(bottom[0]->data_at(0,bottom[0]->channels()-1,0,0)-bottom[0]->data_at(i,bottom[0]->channels()-1,0,0), top[0]->data_at(i, bottom[0]->channels()-1,0,0));
    CHECK_EQ(bottom[0]->data_at(bottom[0]->num()-1,0,0,0)-bottom[0]->data_at(i,0,0,0), top[0]->data_at((bottom[0]->num()-1)*bottom[0]->num()+i,0,0,0));
    CHECK_EQ(bottom[0]->data_at(bottom[0]->num()-1,bottom[0]->channels()-1,0,0)-bottom[0]->data_at(i,bottom[0]->channels()-1,0,0), top[0]->data_at((bottom[0]->num()-1)*bottom[0]->num()+i,bottom[0]->channels()-1,0,0));
  }
  //end of test code*/
   
}

template <typename Dtype>
void CrossEltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	//LOG(FATAL)<<"in here";
	const int dim =bottom[0]->count()/bottom[0]->num();
  const Dtype* top_diff = top[0]->gpu_diff();   //shape:(n*n)*c*w*h
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	Dtype* ones = ones_.mutable_gpu_data();
		for(int k=0; k< bottom[0]->num();k++)
			caffe_gpu_gemv(CblasTrans, bottom[0]->num(), dim, Dtype(1), top_diff+k*bottom[0]->count(), ones,  Dtype(0), bottom_diff+k*dim);
		caffe_gpu_gemv(CblasTrans, bottom[0]->num(),  bottom[0]->count(), Dtype(1), top_diff, ones,  Dtype(0), second_asum_.mutable_gpu_data());
		caffe_gpu_sub(bottom[0]->count(),bottom_diff, second_asum_.mutable_gpu_data(), bottom_diff);
  /*
  //test code
  int testNum=64;
  for(int l=0; l<testNum; l++){
    Dtype res(0.0);
    Dtype last(0.0);
    for(int i=0;i<bottom[0]->num(); i++)	
    {
      res += top[0]->diff_at(l*bottom[0]->num()+i,0,0,0);
      res -= top[0]->diff_at(l+i*bottom[0]->num(),0,0,0);
      last += top[0]->diff_at(l*bottom[0]->num()+i,dim-1,0,0);
      last -= top[0]->diff_at(l+i*bottom[0]->num(),dim-1,0,0);
    }
    if(std::abs(res- bottom[0]->diff_at(l,0,0,0))> 1e-6  && std::abs(last-bottom[0]->diff_at(l,dim-1,0,0)) > 1e-6 )
     	LOG(FATAL)<<"check cross eltwise not passed";
    // CHECK_EQ(res, bottom[0]->diff_at(l,0,0,0));
     //CHECK_EQ(last, bottom[0]->diff_at(l,dim-1,0,0));	
  }
  LOG(INFO)<<"check cross eltwise passed";
  //end of test code
  */
}

INSTANTIATE_LAYER_GPU_FUNCS(CrossEltwiseLayer);


}  // namespace caffe
