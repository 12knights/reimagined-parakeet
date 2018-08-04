#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/combine_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void CombineLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
//	for(int i=0; i<15;i++)  LOG(INFO)<<"bottom[0]: "<<*(bottom[0]->mutable_cpu_data()+i);
//	LOG(INFO)<<"\n";
	int dim = bottom[0]->count()/bottom[0]->num();
	int label;
//	LOG(INFO)<<"bottom_num: "<<bottom[0]->num();
	for (int i = 0; i < bottom[0]->num(); ++i) {
		caffe_gpu_set(dim, Dtype(0.), ave_feature.mutable_gpu_data());
		for(int j=0 ; j<4; j++)
			if(j+i/4*4 != i )
				caffe_gpu_add(dim, ave_feature.mutable_gpu_data(), bottom[0]->mutable_gpu_data()+(j+i/4*4)*dim, ave_feature.mutable_gpu_data() );
		//ave_feature.scale_data(Dtype(1.0/3.0));
		caffe_gpu_scal(dim, Dtype(1.0/3.0), ave_feature.mutable_gpu_data());
		for(int k=0; k<10; k++)
		{
			label =caffe_rng_rand()%5;
			 *(labels.mutable_cpu_data()+(i*10+k)*5+label)=i;
			caffe_gpu_sub(dim, bottom[0]->gpu_data()+i*dim, ave_feature.mutable_gpu_data(), top[0]->mutable_gpu_data()+((i*10+k)*5+label)*dim);
			//LOG(INFO)<<label;
			*(top[1]->mutable_cpu_data()+i*10+k)=label;
			for(int l=0; l<5; l++)
			{
				if (l == label)  continue;
				int  index;
				do{index= caffe_rng_rand()%bottom[0]->num(); }while(index/4 == i/4);
				*(labels.mutable_cpu_data()+(i*10+k)*5+l)=index;
				caffe_gpu_sub(dim, bottom[0]->gpu_data()+index*dim, ave_feature.mutable_gpu_data(), top[0]->mutable_gpu_data()+((i*10+k)*5+l)*dim);
			}
		}
	}
	// 2-norm
	caffe_copy(top[0]->count(),top[0]->gpu_data(), p_or_n.mutable_gpu_data());
	// p_or_n.scale_data(Dtype(2.0));
	caffe_gpu_scal(p_or_n.count(), Dtype(2.0), p_or_n.mutable_gpu_data());
	caffe_gpu_mul(top[0]->count(),top[0]->gpu_data(),top[0]->gpu_data(), top[0]->mutable_gpu_data());
/*	caffe_copy(top[0]->count(), top[0]->mutable_gpu_data(), p_or_n.mutable_gpu_data());
	p_or_n.scale_data(Dtype(-1.0));
	caffe_gpu_exp(top[0]->count() , p_or_n.mutable_gpu_data(), p_or_n.mutable_gpu_data());
	caffe_gpu_add_scalar(top[0]->count() , Dtype(1), p_or_n.mutable_gpu_data());
	caffe_gpu_div(top[0]->count(), ones.mutable_gpu_data(), p_or_n.mutable_gpu_data(), p_or_n.mutable_gpu_data());
	p_or_n.scale_data(Dtype(2.0));
	caffe_gpu_add_scalar(top[0]->count() , Dtype(-1),  p_or_n.mutable_gpu_data());
*/
	//1-norm
//	caffe_gpu_signbit(top[0]->count(),top[0]->gpu_data(), p_or_n.mutable_gpu_data());	
//	caffe_gpu_abs(top[0]->count(), top[0]->gpu_data(), top[0]->mutable_gpu_data());

//	LOG(INFO)<<"top_shape"<<top[0]->num()<<", "<<top[0]->channels();
//	LOG(INFO)<<"top_label"<<
//	for(int i=0; i<bottom[0]->num();i++)  LOG(INFO)<<"label: "<<*(labels.mutable_cpu_data()+i);
//	for(int i=0; i<15;i++)  LOG(INFO)<<"top[1]: "<<*(top[0]->mutable_cpu_data()+i);
//	LOG(INFO)<<"segment line-----------------------------------------\n\n";
}

template <typename Dtype>
void CombineLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const int dim =bottom[0]->count()/bottom[0]->num();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	Dtype* top_diff= top[0]->mutable_gpu_diff();
//	 for(int i=0; i<15;i++)  LOG(INFO)<<"top[0]->diff: "<<*(top[0]->mutable_cpu_diff()+i);
//	LOG(INFO)<<"\n";
	for(int i=0; i<bottom[0]->num()*50; i++)
	{
		int index=*(labels.mutable_cpu_data()+i);
		//caffe_copy(dim, p_or_n.mutable_cpu_data()+i*dim, feature_vec.mutable_cpu_data());  //feature_vec saves p_or_n;
		caffe_gpu_mul(dim, top_diff+i*dim, p_or_n.mutable_gpu_data()+i*dim, feature_vec.mutable_gpu_data());
/*		 for(int iii =0 ; iii<10; iii++)   //passed
                        {
                                         
                                         Dtype a = *(top[0]->mutable_cpu_diff()+i*dim+iii);
                                      //  LOG(INFO)<<"a:"<<a;i
					Dtype b= *(feature_vec.mutable_cpu_data()+iii);
                                        Dtype  c= *(p_or_n.mutable_cpu_data()+i*dim+iii);//bottom[0]->diff_at(k,iii,0,0);
                                        Dtype temp = abs(a*c-b); 
                                        if(  temp> 1e-8 )
                         //                LOG(INFO)<<"temp : "<<temp;
                                                LOG(INFO)<<"a*c:"<<a*c<<", b:"<<b<<", iii:"<<iii;
                        }
*/	
//		caffe_copy(dim,  bottom[0]->gpu_diff()+index*dim, test.mutable_gpu_data());
		caffe_gpu_add(dim, bottom[0]->gpu_diff()+index*dim, feature_vec.gpu_data(), bottom[0]->mutable_gpu_diff()+index*dim);
/*		 for(int iii =0 ; iii<100; iii++)   //passed
                        {

                                         Dtype c = *(bottom[0]->mutable_cpu_diff()+index*dim+iii);
                                      //  LOG(INFO)<<"a:"<<a;i
                                        Dtype b= *(feature_vec.mutable_cpu_data()+iii);
                                        Dtype  a= *(test.mutable_cpu_data()+iii);//bottom[0]->diff_at(k,iii,0,0);
                                        Dtype temp = abs(a+b-c);
                                        if(  temp> 1e-7 )
                         //                LOG(INFO)<<"temp : "<<temp;
                                                LOG(INFO)<<"a:"<<a<<", b:"<<b<<", c:"<<c<<", temp:"<<temp;
                        }
*/
//	caffe_copy(dim, feature_vec.mutable_gpu_data(),  feature_vec.mutable_gpu_diff());	
	//feature_vec.scale_data(Dtype(-1.0/3.0));
	caffe_gpu_scal(dim, Dtype(-1.0/3.0), feature_vec.mutable_gpu_data());
	 //test  scale_data   passed
 /*       for(int iii =0 ; iii<15; iii++)
        {
                for(int jjj=0; jjj<15; jjj++)
                {
			Dtype temp = *(feature_vec.mutable_cpu_diff() + iii + jjj)/Dtype(3.0) + *(feature_vec.mutable_cpu_data() + iii + jjj);
                        if( temp  >1e-7 )
                                LOG(INFO)<<"scale error"<<temp;
                }
        }

*/
			for(int k=(i/50)/4*4; k<(i/50)/4*4+4; k++ ){
				if( k== i/50 ) continue;
//				caffe_copy(dim,  bottom[0]->gpu_diff()+k*dim, test.mutable_gpu_data());
				caffe_gpu_add(dim, bottom[0]->gpu_diff()+k*dim, feature_vec.mutable_gpu_data(),  bottom[0]->mutable_gpu_diff()+k*dim);
				// test add pased
                     /*   for(int iii =0 ; iii<15; iii++)
                        {
                                         Dtype a = test.data_at(iii,0,0,0);
                                        Dtype  b= feature_vec.data_at(iii ,0,0,0);
                                        Dtype  c= bottom[0]->diff_at(k,iii,0,0);
                                        Dtype temp = abs(a+b-c);
                                        if(  temp >  1e-9  )
                                        { 
					LOG(INFO)<<"temp : "<<temp;
                                        LOG(INFO)<<"a:"<<a<<", b:"<<b<<", c:"<<c;
                        		}

			}
		*/
//		for(int iii=0; iii<15;iii++)  LOG(INFO)<<"bottom[0]->diff: "<<*(bottom[0]->mutable_cpu_diff()+iii+0*dim);
  //                      LOG(INFO)<<"segment line-----------------------------------------\n\n";
		}
	}
//	bottom[0]->scale_diff(Dtype(0.2));
//	for(int i=0; i<1500;i++)  LOG(INFO)<<"bottom[0]->diff: "<<*(bottom[0]->mutable_cpu_diff()+i+51*1024);
//        	LOG(INFO)<<"segment line-----------------------------------------\n\n";
}

INSTANTIATE_LAYER_GPU_FUNCS(CombineLayer);


}  // namespace caffe
