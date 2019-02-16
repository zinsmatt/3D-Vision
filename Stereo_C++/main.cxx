#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>


enum class StereoScoringMethod { SSD, SAD };

/**
 * @brief compute_stereo
 * @param image_l [in] left grayscale image
 * @param image_r [in] right grayscale image
 * @param disp_levels [in] number of disparity levels
 * @param block_size [in] size of the block used for matching
 * @param method [in] method used for scoring
 * @param border_size [in] size of the ignored borders
 * @param filter [in] filter disparity
 * @return
 */
cv::Mat compute_stereo(cv::Mat const& image_l, cv::Mat const& image_r,
                       int disp_levels, int block_size, StereoScoringMethod method,
                       int border_size, bool filter)
{
  cv::Mat disp(image_l.rows, image_l.cols, CV_8U, 0.0);
  if (image_l.channels() != 1 || image_r.channels() != 1)
  {
    std::cout << "Input images should be single channel images.\n";
    return disp;
  }

  cv::Mat img_l, img_r;
  image_l.convertTo(img_l, CV_32S);
  image_r.convertTo(img_r, CV_32S);
  if (block_size % 2 == 0)
  {
    std::cout << "Warning block size should be odd. +1 is used\n";
    block_size += 1;
  }
  int block_half_size = block_size /2;
  if (border_size < block_size)
  {
    std::cout << "Warning border size should be at least equal to half block size";
    border_size = block_half_size;
  }

  for (int i = border_size; i < disp.rows - border_size; ++i)
  {
    for (int j = border_size; j < disp.cols - border_size; ++j)
    {
      std::vector<int> scores(disp_levels, std::numeric_limits<int>::max());
      std::vector<int> scores_index(disp_levels);
      std::iota(scores_index.begin(), scores_index.end(), 0);
      const cv::Mat block_r = img_r(cv::Range(i - block_half_size, i + block_half_size + 1),
                                    cv::Range(j - block_half_size, j + block_half_size + 1));

      for (int d = 0; d <= disp_levels && d + j < disp.cols - block_half_size; ++d)
      {
        const cv::Mat block_l = img_l(cv::Range(i - block_half_size, i + block_half_size + 1),
                                      cv::Range(j - block_half_size + d, j + block_half_size + 1 + d));
        int score;
        if (method == StereoScoringMethod::SAD)
        {
          score = cv::sum(cv::abs(block_r - block_l))[0];
        }
        else if (method == StereoScoringMethod::SSD)
        {
          cv::Mat squared;
          cv::pow(block_r - block_l, 2, squared);
          score = cv::sum(squared)[0];
        }
        scores[d] = score;
      }
      // sort scores
      std::sort(scores_index.begin(), scores_index.end(), [&scores] (int i, int j) {
        return scores[i] < scores[j];
      });

      // filter noise
      if (filter && scores[scores_index[0]] == std::numeric_limits<int>::max())
        continue;
      if (filter && scores[scores_index[0]] * 1.5 >= scores[scores_index[1]])
        continue;
      disp.at<uchar>(i, j) = scores_index[0];
    }
  }
  return disp;
}


int main()
{
  cv::Mat image_l = cv::imread("../tsukuba/scene1.row3.col1.ppm");
  cv::Mat image_r = cv::imread("../tsukuba/scene1.row3.col2.ppm");

  // convert to grayscale images
  cv::Mat image_l_grey, image_r_grey;
  cv::cvtColor(image_l, image_l_grey, CV_BGR2GRAY);
  cv::cvtColor(image_r, image_r_grey, CV_BGR2GRAY);

  int block_size = 5;
  int disp_levels = 16;
  int border_size = 16;
  cv::Mat disp = compute_stereo(image_l_grey, image_r_grey, disp_levels, block_size, StereoScoringMethod::SAD, border_size, false);

  disp.convertTo(disp, CV_32F, 1.0 / disp_levels);

  cv::namedWindow("win", cv::WINDOW_NORMAL);
  cv::imshow("win", disp);
  cv::waitKey(0);
  cv::destroyAllWindows();

  disp.convertTo(disp, CV_8U, 255);
  cv::imwrite("disparity.png", disp);

  return 0;
}
