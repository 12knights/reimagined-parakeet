#include <algorithm>
#include <vector>

#include "caffe/layers/cos_dist_layer.hpp"
#include "caffe/util/math_functions.hpp"	

namespace caffe {

template <typename Dtype>
void CosDistLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	ones_.Reshape(bottom[0]->num(),1,1,1);
	ones_dim_.Reshape(bottom[0]->count()/bottom[0]->num(),1,1,1);
	norm_.Reshape(bottom[0]->num(),1,1,1);
	caffe_set(bottom[0]->num(), Dtype(1), ones_.mutable_cpu_data());
	caffe_set(bottom[0]->count()/bottom[0]->num(), Dtype(1), ones_dim_.mutable_cpu_data());
	vector<int> bottom_shape = bottom[0]->shape();	
	bottom_square_.Reshape(bottom_shape);
	bottom_norm_.Reshape(bottom_shape);
	diff1_.Reshape(bottom_shape);
	diff2_.Reshape(bottom_shape);
	bottom_shape[0]=bottom[0]->num()*bottom[0]->num();
	bottom_shape[1]=1;
	bottom_shape[2]=1;
	bottom_shape[3]=1;
	identity_.Reshape(bottom_shape);
	num_num_.Reshape(bottom_shape);
	inner_prod_.Reshape(bottom_shape);
	top[0]->Reshape(bottom_shape);
	caffe_set(bottom[0]->num()*bottom[0]->num(), Dtype(0), identity_.mutable_cpu_data());
	for(int i=0; i<bottom[0]->num(); i++)
	{
		caffe_set(1, Dtype(1.0),identity_.mutable_cpu_data()+bottom[0]->num()*i+i);
	}
}



template <typename Dtype>
void CosDistLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
}


template <typename Dtype>
void CosDistLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const int count = top[0]->count();
	const int num = bottom[0]->num();
	const int dim =bottom[0]->count()/bottom[0]->num();/*
//test 
	for(int i=0; i<num; i++ )
	{
		for(int k=0; k<dim; k++)
			LOG(INFO)<<bottom[0]->data_at(i,k,0,0);
		LOG(INFO)<<"                                      ";
	}
//test end

*/

  Dtype* top_data = top[0]->mutable_cpu_data();  //shape:(n*n)*c*w*h
	const Dtype*  bottom_data=bottom[0]->cpu_data();
  caffe_set(count, Dtype(0.0), top_data);
	caffe_powx(bottom[0]->count(), bottom_data, Dtype(2.0), bottom_square_.mutable_cpu_data());    //caculate l2 norm
	caffe_cpu_gemv(CblasNoTrans, num, dim, Dtype(1.0), bottom_square_.mutable_cpu_data(), ones_dim_.cpu_data(),Dtype(0.0),  norm_.mutable_cpu_data());   
	caffe_sqrt(num, norm_.cpu_data(), norm_.mutable_cpu_data());
  
	for(int k=0; k< num; k++)
	{
		caffe_cpu_axpby(dim, Dtype(1.0)/norm_.data_at(k,0,0,0), bottom_data+k*dim, Dtype(0.0),  bottom_norm_.mutable_cpu_data()+k*dim) ;
	}
	

	caffe_cpu_gemm(CblasNoTrans, CblasTrans, num, dim, num, Dtype(1.0), bottom_norm_.cpu_data(), bottom_norm_.cpu_data(), Dtype(0.0), inner_prod_.mutable_cpu_data());  //caculate innerproduct
	


	for(int k=0; k< num; k++)
	{
		caffe_sub(num, ones_.cpu_data(), inner_prod_.cpu_data()+k*num, top[0]->mutable_cpu_data()+k*num);
	}
	caffe_scal(top[0]->count(), Dtype(0.5), top[0]->mutable_cpu_data());


}

template <typename Dtype>
void CosDistLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	//LOG(FATAL)<<"in here";
	const int dim =bottom[0]->count()/bottom[0]->num();
	const int num =bottom[0]->num();
	const int count =bottom[0]->count();
	caffe_copy(num*num, top[0]->cpu_diff(),num_num_.mutable_cpu_data());
	
	caffe_cpu_gemm(CblasNoTrans, CblasTrans, num, num, num, Dtype(1.0), identity_.cpu_data(), top[0]->cpu_diff(), Dtype(0.0), num_num_.mutable_cpu_data());
	caffe_add(num*num, num_num_.cpu_data(), top[0]->cpu_diff(), num_num_.mutable_cpu_data());
	
	



	for(int i=0; i<num; i++) 
	{
		caffe_copy(count, bottom_norm_.cpu_data(), diff1_.mutable_cpu_data());   //diff1_.data for yl/|yl|


		for(int d=0; d<num; d++)
		{
			caffe_copy(dim, diff1_.mutable_cpu_data()+i*dim, diff2_.mutable_cpu_data()+d*dim);    //diff2_.data for top(k,l)*xk/|xk|
			caffe_scal(dim, inner_prod_.data_at(i*num+d,0,0,0), diff2_.mutable_cpu_data()+d*dim);
		}

		caffe_sub(count, diff2_.cpu_data(),diff1_.cpu_data(),  diff2_.mutable_cpu_data());
		caffe_scal(count, Dtype(1.0)/norm_.data_at(i,0,0,0), diff2_.mutable_cpu_data());

		for(int l=0; l<num; l++)
		{
			caffe_scal(dim, num_num_.data_at(i*num+l,0,0,0), diff2_.mutable_cpu_data());
		}

		caffe_cpu_gemv(CblasTrans, num, dim, Dtype(1.0), diff2_.mutable_cpu_data(), ones_.cpu_data(), Dtype(0.0),  bottom[0]->mutable_cpu_diff()+i*dim);   
	}
	caffe_scal(bottom[0]->count(), Dtype(0.5), bottom[0]->mutable_cpu_diff());
	


}
#ifdef CPU_ONLY
STUB_GPU(CosDistLayer);
#endif

INSTANTIATE_CLASS(CosDistLayer);
REGISTER_LAYER_CLASS(CosDist);
}  // namespace caffe
