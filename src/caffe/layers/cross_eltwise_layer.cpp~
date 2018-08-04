#include <algorithm>
#include <vector>

#include "caffe/layers/cross_eltwise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CrossEltwiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	second_asum_.Reshape(bottom[0]->shape());
	ones_.Reshape(bottom[0]->num(),1,1,1);
	caffe_set(bottom[0]->num(), Dtype(1), ones_.mutable_cpu_data());
	vector<int> top_shape = bottom[0]->shape();	
	top_shape[0]=bottom[0]->num()*bottom[0]->num();
	top[0]->Reshape(top_shape);
	top[1]->Reshape(top[0]->num(),1,1,1);
}



template <typename Dtype>
void CrossEltwiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
   
  }
  
}


template <typename Dtype>
void CrossEltwiseLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
   const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();  //shape:(n*n)*c*w*h
  caffe_set(count, Dtype(0.), top_data);
  // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
	int dim =bottom[0]->count()/bottom[0]->num();
  for(int k=0; k< bottom[0]->num(); k++)
	{
		for(int i=0; i<bottom[0]->num(); i++)	
			caffe_copy(dim, bottom[0]->cpu_data()+k*dim, top_data+k*bottom[0]->count()+i*dim);
		//caffe_copy(bottom[0]->count(),bottom[0]->gpu_data(), top_data+k*bottom[0]->count() );
	   caffe_cpu_axpby(bottom[0]->count(), Dtype(-1), bottom[0]->cpu_data(), Dtype(1), top_data+k*bottom[0]->count());
	}
}

template <typename Dtype>
void CrossEltwiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const int dim =bottom[0]->count()/bottom[0]->num();
  const Dtype* top_diff = top[0]->cpu_diff();   //shape:(n*n)*c*w*h
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	Dtype* ones = ones_.mutable_cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
				
				for(int k=0; k< bottom[0]->num();k++)
					caffe_cpu_gemv(CblasTrans, bottom[0]->num(), dim, Dtype(1), top_diff+k*bottom[0]->count(), ones,  Dtype(0), bottom_diff+k*dim);
				caffe_cpu_gemv(CblasTrans, bottom[0]->num(),  bottom[0]->count(), Dtype(1), top_diff, ones,  Dtype(0), second_asum_.mutable_cpu_data());
				caffe_sub(bottom[0]->count(),bottom_diff, second_asum_.mutable_cpu_data(), bottom_diff);			
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(CrossEltwiseLayer);
#endif

INSTANTIATE_CLASS(CrossEltwiseLayer);
REGISTER_LAYER_CLASS(CrossEltwise);
}  // namespace caffe
