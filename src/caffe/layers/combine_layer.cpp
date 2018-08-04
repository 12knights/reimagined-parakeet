#include <algorithm>
#include <vector>

#include "caffe/layers/combine_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CombineLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	vector<int> bottom_shape = bottom[0]->shape();
	vector<int> top_shape = bottom[0]->shape();
	top_shape[1]=bottom[0]->channels()*5;
	top_shape[0]=bottom[0]->num()*10;
	top[0]->Reshape(top_shape);
	ones.Reshape(top_shape);
	p_or_n.Reshape(top_shape);
	caffe_set(top[0]->count(),  Dtype(1.0), ones.mutable_cpu_data());
	top[1]->Reshape(bottom[0]->num()*10,1,1,1);
	labels.Reshape(bottom[0]->num()*10*5,1,1,1);	
	feature_vec.Reshape(bottom[0]->count()/bottom[0]->num(),1,1,1);
	ave_feature.Reshape(bottom[0]->count()/bottom[0]->num(),1,1,1);
	test.Reshape(bottom[0]->count()/bottom[0]->num(),1,1,1);	
}



template <typename Dtype>
void CombineLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

// labels is a num*10*5 array, containing which picture it is at any position
//p_or_n is a num*10*5*dim array, containing whether it is positive or negative at any point
template <typename Dtype>
void CombineLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	Dtype* bottom_data= bottom[0]->mutable_cpu_data();
	Dtype* top_data= top[0]->mutable_cpu_data();
	int dim = bottom[0]->count()/bottom[0]->num();
	int label;
	for (int i = 0; i < bottom[0]->num(); ++i) {
		caffe_set(dim, Dtype(0.), ave_feature.mutable_cpu_data());
		for(int j=0 ; j<4; j++)
			if(j+i/4*4 != i )
				caffe_add(dim, ave_feature.mutable_cpu_data(), bottom_data+(j+i/4*4)*dim, ave_feature.mutable_cpu_data() );
		for(int k=0; k<10; k++)
		{
			label = caffe_rng_rand()%5;
			*(labels.mutable_cpu_data()+(i*10+k)*5+label)=i;
			caffe_sub(dim, bottom_data+i*dim, ave_feature.mutable_cpu_data(), top_data+((i*10+k)*5+label)*dim);
			*(top[1]->mutable_cpu_data()+i*10+k)=label;
			for(int l=0; l<5; l++)
			{
				if (l == label)  continue;
				int  index;
				do{index= caffe_rng_rand()%bottom[0]->num();}while(index/4!=i);
				*(labels.mutable_cpu_data()+(i*10+k)*5+l)=index;
				caffe_sub(dim, bottom_data+index*dim, ave_feature.mutable_cpu_data(), top_data+((i*10+k)*5+l)*dim);
			}
		}
	}
	caffe_signbit(bottom[0]->count(),top_data, p_or_n.mutable_cpu_data());
	caffe_abs(bottom[0]->count(),top_data, top_data);
}

template <typename Dtype>
void CombineLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const int dim =bottom[0]->count()/bottom[0]->num();
	Dtype* bottom_diff= bottom[0]->mutable_cpu_diff();
	Dtype* top_diff= top[0]->mutable_cpu_diff();
	for(int i=0; i<bottom[0]->num()*50; i++)
	{
		int index=*(labels.mutable_cpu_data()+i);
		//caffe_copy(dim, p_or_n.mutable_cpu_data()+i*dim, feature_vec.mutable_cpu_data());  //feature_vec saves p_or_n;
		caffe_mul(dim, top_diff+i*dim, p_or_n.mutable_cpu_data()+i*dim, feature_vec.mutable_cpu_data());
		caffe_add(dim, bottom_diff+index*dim, feature_vec.mutable_cpu_data(),  bottom_diff+index*dim);
		feature_vec.scale_data(-1/3);
		for(int k=(i/50)/4*4; k<(i/50)/4*4+4;k++)
		{
			if(k== i/50) continue;
			caffe_add(dim, bottom_diff+k*dim, feature_vec.mutable_cpu_data(),  bottom_diff+k*dim);
		}
	}
}


#ifdef CPU_ONLY
STUB_GPU(CombineLayer);
#endif

INSTANTIATE_CLASS(CombineLayer);
REGISTER_LAYER_CLASS(Combine);
}  // namespace caffe
