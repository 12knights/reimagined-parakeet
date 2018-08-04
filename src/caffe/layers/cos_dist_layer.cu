#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/cos_dist_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void CosDistLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const int count = top[0]->count();
	const int num = bottom[0]->num();
	const int dim =bottom[0]->count()/bottom[0]->num();
/*
//test 
	for(int i=0; i<num; i++ )
	{
		for(int k=0; k<dim; k++)
			LOG(INFO)<<bottom[0]->data_at(i,k,0,0);
		LOG(INFO)<<"                                      ";
	}
//test end
*/
  Dtype* top_data = top[0]->mutable_gpu_data();  //shape:(n*n)*c*w*h
	const Dtype*  bottom_data=bottom[0]->gpu_data();
  caffe_gpu_set(count, Dtype(0.0), top_data);
	caffe_gpu_powx(bottom[0]->count(), bottom_data, Dtype(2.0), bottom_square_.mutable_gpu_data());    //caculate l2 norm
	caffe_gpu_gemv(CblasNoTrans, num, dim, Dtype(1.0), bottom_square_.mutable_gpu_data(), ones_dim_.gpu_data(),Dtype(0.0),  norm_.mutable_gpu_data());   
	caffe_gpu_sqrt(num, norm_.gpu_data(), norm_.mutable_gpu_data());
  /*	//check  norm_  , passed
	for(int i=0; i<num; i++)
	{	
		Dtype n_=0;
		for(int k=0; k<dim; k++)
		{
			n_ += bottom[0]->data_at(i,k,0,0)*bottom[0]->data_at(i,k,0,0);
		}
		n_=std::sqrt(n_);
		CHECK_LE(std::abs(norm_.data_at(i,0,0,0)-n_)/n_,1e-6);
	}
	//check end
	*/
	for(int k=0; k< num; k++)
	{
		caffe_gpu_axpby(dim, Dtype(1.0)/norm_.data_at(k,0,0,0), bottom_data+k*dim, Dtype(0.0),  bottom_norm_.mutable_gpu_data()+k*dim) ;
	}
	/*//check  bottom_norm_, passed
	for(int i=0; i<num; i++)
	{
		for(int k=0; k<dim; k++)
		{
			Dtype checkNum=bottom[0]->data_at(i,k,0,0)/norm_.data_at(i,0,0,0);
			//LOG(INFO)<<"checkNum:"<<checkNum<<"bottom_norm_.data_at(i,k,0,0):"<<bottom_norm_.data_at(i,k,0,0);
			CHECK_LE(std::abs(checkNum-bottom_norm_.data_at(i,k,0,0)),1e-5)<<"checkNum:"<<checkNum<<"bottom_norm_.data_at(i,k,0,0):"<<bottom_norm_.data_at(i,k,0,0);
		}
	}
	//check end*/

	caffe_gpu_gemm(CblasNoTrans, CblasTrans, num, num, dim, Dtype(1.0), bottom_norm_.gpu_data(), bottom_norm_.gpu_data(), Dtype(0.0), inner_prod_.mutable_gpu_data());  //caculate innerproduct

	/*  //test for inner_prod_, passed ; ones_ passed ;
	for(int j=2; j<num; j++){
		Dtype res =0;
	for(int i=0; i<dim; i++)
	{
		res += bottom_norm_.data_at(j-1, i,0,0)*bottom_norm_.data_at(j-2, i,0,0);
	}
	CHECK_LE(std::abs(   res-inner_prod_.data_at((j-1)*num+j-2,0,0,0)  ) ,  1e-6);
}

	for(int i=0; i<num; i++)
	{
		CHECK_LE(std::abs(ones_.data_at(i,0,0,0)-Dtype(1)),1e-6);
	}
*/
	for(int k=0; k< num; k++)
	{
		caffe_gpu_sub(num, ones_.gpu_data(), inner_prod_.gpu_data()+k*num, top[0]->mutable_gpu_data()+k*num);
	}
	/*  //1-inner_prod_, passed;
	for(int i=1; i<num; i++)
	{
		CHECK_LE(std::abs(1-inner_prod_.data_at(i*num+i-1,0,0,0)-top[0]->data_at(i*num+i-1,0,0,0)),1e-6);
	}

*/
	caffe_gpu_scal(top[0]->count(), Dtype(0.5), top[0]->mutable_gpu_data());
	for(int i=0; i<num; i++)
	{
		caffe_gpu_set(1, Dtype(0.0), top[0]->mutable_gpu_data()+i*num+i);
	}
	caffe_gpu_sqrt(top[0]->count(),  top[0]->gpu_data(), top[0]->mutable_gpu_data());
	//LOG(INFO)<<"top[0]->data_at(10*num+10,0,0,0):"<<top[0]->data_at(k*num+l,0,0,0)<<",,,test:"<<test;
	//CHECK_LE(std::abs(std::sqrt(test)- top[0]->data_at(k*num+l,0,0,0)),1e-9);
/*  //test for top[0], passed
	for(int i=1; i<num; i++)
	{
		CHECK_LE(std::abs((1-inner_prod_.data_at(i*num+i-1,0,0,0))*Dtype(0.5)-top[0]->data_at(i*num+i-1,0,0,0)),1e-6);
	}
*/
}






template <typename Dtype>
void CosDistLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	//LOG(FATAL)<<"in here";
	const int dim =bottom[0]->count()/bottom[0]->num();
	const int num =bottom[0]->num();
	const int count =bottom[0]->count();
	/*
	for(int i=0; i<1; i++ )
	{
		for(int k=0; k<num; k++)
			LOG(INFO)<<top[0]->diff_at(i*num+k,0,0,0);
		LOG(INFO)<<"                                      ";
	}
*/
	caffe_copy(num*num, top[0]->gpu_diff(),num_num_.mutable_gpu_data());
	//caffe_gpu_set(num*num, Dtype(0.0), num_num_.mutable_gpu_data());
	caffe_gpu_gemm(CblasNoTrans, CblasTrans, num, num, num, Dtype(1.0), identity_.gpu_data(), top[0]->gpu_diff(), Dtype(1.0), num_num_.mutable_gpu_data());
	/*
	for(int i=0; i<num; i++ )
	{
		for(int k=0; k<num; k++)
			caffe_gpu_add(1, num_num_.gpu_data()+i*num+k, num_num_.gpu_data()+k*num+i, num_num_.mutable_gpu_data()+k*num+i );
	}
*/
/* // check for num_num_, passed;
	for(int i=0; i<num; i++ )
	{
		for(int k=0; k<num; k++){
			CHECK_LE(std::abs(top[0]->diff_at(i*num+k,0,0,0)+top[0]->diff_at(k*num+i,0,0,0)-num_num_.data_at(i*num+k,0,0,0)),1e-6);
			CHECK_LE(std::abs(top[0]->diff_at(i*num+k,0,0,0)+top[0]->diff_at(k*num+i,0,0,0)-num_num_.data_at(k*num+i,0,0,0)),1e-6);
		}
	}
 //check end
*/
	for(int i=0; i<num; i++) 
	{
		caffe_copy(count, bottom_norm_.gpu_data(), diff1_.mutable_gpu_data());   //diff1_.data for yl/|yl|
		for(int d=0; d<num; d++)
		{
			caffe_copy(dim, diff1_.gpu_data()+i*dim, diff2_.mutable_gpu_data()+d*dim);    //diff2_.data for inner_prod_(k,l)*xk/|xk|
			caffe_gpu_scal(dim, inner_prod_.data_at(i*num+d,0,0,0), diff2_.mutable_gpu_data()+d*dim);
		}
		caffe_gpu_sub(count, diff2_.gpu_data(),diff1_.gpu_data(),  diff2_.mutable_gpu_data());
		caffe_gpu_scal(count, Dtype(1.0)/norm_.data_at(i,0,0,0), diff2_.mutable_gpu_data());


		
		for(int l=0; l<num; l++)
		{
			Dtype scal(0);
			if(i==l)
				CHECK_LE(top[0]->data_at(i*num+l,0,0,0),1e-5);
			if(top[0]->data_at(i*num+l,0,0,0)>1e-4)
				scal=num_num_.data_at(i*num+l,0,0,0)*Dtype(0.25)/top[0]->data_at(i*num+l,0,0,0);
			//LOG(INFO)<<"scal:"<<scal;
			caffe_gpu_scal(dim, scal , diff2_.mutable_gpu_data());
		}
		caffe_gpu_gemv(CblasTrans, num, dim, Dtype(1.0), diff2_.mutable_gpu_data(), ones_.gpu_data(), Dtype(0.0),  bottom[0]->mutable_gpu_diff()+i*dim);   

	/*  //check for sum ; passed
		for (int ll =0 ; ll<dim; ll++)
		{
			Dtype res=0;
			for(int kk =0; kk<num; kk++)	
			{
				res += diff2_.data_at(kk,ll,0,0);
			}
			CHECK_LE(std::abs(res-bottom[0]->diff_at(i,ll,0,0)), 1e-6);
		}

		*/
	}
	//caffe_gpu_scal(bottom[0]->count(), Dtype(0.5), bottom[0]->mutable_gpu_diff());
	/*
	for(int i=16; i<17; i++ )
	{
		for(int k=0; k<5; k++)
			LOG(INFO)<<"bottom[0]->diff_at(i,k,0,0)"<<bottom[0]->diff_at(i,k,0,0);
		LOG(INFO)<<"                                      ";
	}
	*/
}

INSTANTIATE_LAYER_GPU_FUNCS(CosDistLayer);


}  // namespace caffe
