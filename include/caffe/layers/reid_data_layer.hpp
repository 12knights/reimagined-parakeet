#ifndef CAFFE_REID_DATA_LAYER_HPP_
#define CAFFE_REID_DATA_LAYER_HPP_
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <vector>
#include <cfloat>
#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/rng.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
class ReidDataLayer : public ReidPrefetchingDataLayer<Dtype> {
 public:
  explicit ReidDataLayer(const LayerParameter& param)
      : ReidPrefetchingDataLayer<Dtype>(param) {}
  virtual ~ReidDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ReidData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(ReidBatch<Dtype>* batch);
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual unsigned int RandRng();

   inline vector<size_t> batch_ids() {
    const int batch_size = this->layer_param_.reid_data_param().batch_size();
    const size_t total = this->lines_.size();
    vector<size_t> ans;
    for (int _ = 0; _ < batch_size;) {
      size_t cur_id = RandRng() % total;
			
			// add the same label
			int same_num = this->layer_param_.reid_data_param().same_num();
			int cur_label = this->lines_[cur_id].second;
			const vector<size_t>& sets = this->label_set[cur_label];
			int choose_num=std::min(int(sets.size()/2+1),int(same_num));
			size_t *choose=new size_t[choose_num];
			int num=0;
			while(num<choose_num)
			{
				int index=RandRng() % (sets.size()-1);
				int already_in=0;
				for(int kk=0; kk<num; kk++)
				{
					if(choose[kk]==index){already_in=1; break;}
				}
				if(already_in == 0){choose[num++]=sets[index];}
			}
			for(int i=0 ; i< choose_num; i++)
			{
				size_t next_id = choose[i];
				if( _ < batch_size)
				{
					ans.push_back( next_id);
					_++;
				}
				else
					break;
			}
			delete []choose;
    }
		
    return ans;
  }
  inline size_t pair_label(size_t label) {
    CHECK(label >= 0 && label < label_set.size());
    if (--left_images <= 0) {
      left_images = this->lines_.size();
      this->pos_fraction = std::min(this->pos_fraction*this->layer_param_.reid_data_param().pos_factor(), Dtype(this->layer_param_.reid_data_param().pos_limit()));
      this->neg_fraction = std::min(this->neg_fraction*this->layer_param_.reid_data_param().neg_factor(), Dtype(this->layer_param_.reid_data_param().neg_limit()));
      DLOG(INFO) << "Epoch done, pos_fraction : " << this->pos_fraction << ", neg_fraction : " << this->neg_fraction;
    }
    const size_t pos = this->pos_fraction * 10000;
    const size_t neg = this->neg_fraction * 10000;
    const size_t x = RandRng() % (pos + neg);
    size_t cor_label = INT_MAX;
    if (x < pos) {
      cor_label = label;
    } else {
      cor_label = RandRng() % (label_set.size()-1);
      if (cor_label >= label) cor_label ++;
    }
    DLOG(INFO) << "Reid Data Layer pair_label (PN) : " << pos << " : " << neg 
        << " x : " << x << " ||| " << cor_label;
    const vector<size_t>& sets = label_set[cor_label];
    return sets[ RandRng() % sets.size() ];
  }

  inline vector<size_t> batch_pairs(vector<size_t> ids) {
    const int batch_size = this->layer_param_.reid_data_param().batch_size();
    CHECK_EQ(ids.size(), batch_size);
    vector<size_t> ans;
    for (int idx = 0; idx < batch_size; idx++) {
      size_t image_idx = ids[idx];
      ans.push_back( pair_label(this->lines_[image_idx].second) );
    }
    return ans;
  }

  vector<std::pair<std::string, int> > lines_;
  vector<cv::Mat> cv_imgs_;
  vector<vector<size_t> > label_set;
  Dtype pos_fraction;
  Dtype neg_fraction;
  int left_images;
};

}  // namespace caffe

#endif  // CAFFE_REID_DATA_LAYER_HPP_
