#include "slam.h"

#include <sys/prctl.h>

#include "rtkm.h"
#include "floam.h"
#include "gicpm.h"
#include "fastlio.h"
#include "localization.h"
#include "backend_api.h"

#include "Utils.h"
#include "InterProcess.h"
#include "UDPServer.h"

SLAM::runType SLAM::getRunModeType(std::string name) {
  if (name.compare("online") == 0) {
    return runType::Online;
  } else {
    return runType::Offline;
  }
}

SLAM::modeType SLAM::getMappingTypeByName(std::string name) {
  if (name.compare("RTKM") == 0) {
    return modeType::RTKM;
  } else if (name.compare("FLOAM") == 0) {
    return modeType::FLOAM;
  } else if (name.compare("GICPM") == 0) {
    return modeType::GICPM;
  } else if (name.compare("FastLIO") == 0) {
    return modeType::FastLIO;
  } else if (name.compare("Localization") == 0) {
    return modeType::Localization;
  } else {
    return modeType::None;
  }
}

std::string SLAM::getMappingNameByType(modeType type) {
  switch (type) {
    case modeType::RTKM:
      return "RTKM";
    case modeType::FLOAM:
      return "FLOAM";
    case modeType::GICPM:
      return "GICPM";
    case modeType::FastLIO:
      return "FastLIO";
    case modeType::Localization:
      return "Localization";
    default:
      return "";
  }
}

SLAM::SLAM(enum runType run, modeType modeIn) : mRunMode(run), mMappingMode(modeIn) {
  if (mMappingMode == modeType::RTKM) {
    mSlam.reset(new Mapping::RTKM());
  } else if (mMappingMode == modeType::FLOAM) {
    mSlam.reset(new Mapping::HDL_FLOAM());
  } else if (mMappingMode == modeType::GICPM) {
    mSlam.reset(new Mapping::HDL_GICP());
  } else if (mMappingMode == modeType::FastLIO) {
    mSlam.reset(new Mapping::HDL_FastLIO());
  } else if (mMappingMode == modeType::Localization) {
    mSlam.reset(new Locate::Localization());
  } else {
    LOG_ERROR("Unknown SLAM type: {}", mMappingMode);
  }

  if (mMappingMode != modeType::Localization) {
    Locate::Localization::releaseStaticResources();
  }

  mProjector.reset(new UTMProjector());
  mUnixServer.reset(new UnixSocketServer("/tmp/imu_data.sock"));

  mUseGPS = false;
  mUseIMU = false;
  mInsDim = 2;
}

SLAM::~SLAM() {
  if (mMappingThread != nullptr) {
    mThreadStart = false;
    mMappingThread->join();
    mMappingThread.reset(nullptr);
  }
  if (mLocalizationThread != nullptr) {
    mThreadStart = false;
    mLocalizationThread->join();
    mLocalizationThread.reset(nullptr);
  }
  mSlam.reset(nullptr);
}

std::vector<std::string> SLAM::setSensors(std::vector<std::string> sensors) {
  std::vector<std::string> sensor_list = mSlam->setSensors(sensors);
  for (auto sensor : sensor_list) {
    if (sensor.compare("RTK") == 0) {
      mUseGPS = true;
    } else if (sensor.compare("IMU") == 0) {
      mUseIMU = true;
    }
  }
  return sensor_list;
}

void SLAM::setParams(const std::string& map_path, const double resolution,
                     float dist_threshold, float degree_threshold, float frame_range) {
  mInitParameter.map_path = map_path;
  mInitParameter.resolution = resolution;
  mInitParameter.key_frame_distance = dist_threshold;
  mInitParameter.key_frame_degree = degree_threshold;
  mInitParameter.key_frame_range = frame_range;
  mInitParameter.scan_period = 0.1; // assume 10 Hz

  pcl::RadiusOutlierRemoval<Point>::Ptr rad(new pcl::RadiusOutlierRemoval<Point>());
  rad->setRadiusSearch(1.0);
  rad->setMinNeighborsInRadius(3);
  mOutlierRemoval = rad;
}

void SLAM::setCameraParameter(const std::map<std::string, CamParamType>& camParam) {
  mSlam->setCameraParameter(camParam);
}

void SLAM::setStaticParameter(const Eigen::Matrix4d& trans) {
  mStaticTrans = trans;
  mSlam->setStaticTransform(trans);
}

void SLAM::setImuStaticParameter(const Eigen::Matrix4d& trans) {
  mImuStaticTrans = trans;
  mSlam->setImuStaticTransform(mImuStaticTrans);
}

void SLAM::setInsConfig(const std::map<int, InsConfig> &config) {
  mInsConfig = config;
}

void SLAM::setInsDimension(int dim) {
  mInsDim = dim;
}

void SLAM::setInitPose(const Eigen::Matrix4d &t) {
  mSlam->setInitPose(t);
}

int SLAM::getEstimatePose(double x0, double y0, double x1, double y1, Eigen::Matrix4d &init_guess) {
  Eigen::Vector3d v0(x1 - x0, y1 - y0, 0);
  Eigen::Vector3d v1(1, 0, 0);
  v0.normalize();

  init_guess = Eigen::Matrix4d::Identity();
  init_guess.topRightCorner<3, 1>() = Eigen::Vector3d(x0, y0, 0);
  init_guess.topLeftCorner<3, 3>() = Eigen::Matrix3d(Eigen::Quaterniond::FromTwoVectors(v1, v0));
  return mSlam->getEstimatePose(init_guess);
}

void SLAM::setDestination(bool enable, std::string dest, int port) {
  mOutput = enable;
  mDestinationIP = dest;
  mDestinationPort = port;
}

bool SLAM::setup() {
  mLastInsPriority = -1;
  mLastInsTimestamp = 0;
  bool result = mSlam->init(mInitParameter);

  // compute the map origin
  if (mSlam->originIsSet()) {
    const RTKType& origin = mSlam->getOrigin();
    Eigen::Vector3d zero_xyz;
    mProjector->FromGlobalToLocal(origin.latitude, origin.longitude, zero_xyz(0), zero_xyz(1));
    zero_xyz(2) = origin.altitude;
    mZeroUtm = zero_xyz;
  }

  if (mMappingMode != modeType::Localization) {
    mThreadStart = true;
    mMappingThread.reset(new std::thread(&SLAM::runMappingThread, this));
  } else {
    mThreadStart = true;
    mLocalizationThread.reset(new std::thread(&SLAM::runLocalizationThread, this));
  }
  return result;
}

bool SLAM::getKeyframe(PointCloudAttrImagePose &keyframe) {
  return mKeyframeQueue.try_dequeue(keyframe);
}

void SLAM::getGraphMap(std::vector<std::shared_ptr<KeyFrame>> &frames) {
  return mSlam->getGraphMap(frames);
}

void SLAM::getColorMap(PointCloudRGB::Ptr &points) {
  mSlam->getColorMap(points);
}

bool SLAM::preprocessInsData(std::shared_ptr<RTKType> &rtk) {
  // check the valid of INS data
  if (rtk->status == 0 && fabs(rtk->longitude) < 1e-4 && fabs(rtk->latitude) < 1e-4) {
    double lost_time = getCurrentTime() / 1000000.0 - mLastInsTimestamp;
    if (mLastInsPriority != -1 && lost_time >= 1.0) {
      LOG_WARN("SLAM: rtk status downgrade from {} to invalid", mInsConfig[mLastInsPriority].status_name);
      mLastInsPriority = -1;
      mLastInsTimestamp = 0;
    }
    return false;
  }

  // don't check the status when doing RTK Mapping
  if (mMappingMode == modeType::RTKM) {
    return true;
  }

  // to find a matching config
  InsConfig match_config;
  match_config.priority = -1;
  for (auto &config : mInsConfig) {
    if (config.second.status == rtk->status) {
      match_config = config.second;
      break;
    }
  }

  // if no match, check if "any" status is set
  if (match_config.priority == -1) {
    for (auto &config : mInsConfig) {
      if (config.second.status == -1) {
        match_config = config.second;
        break;
      }
    }
  }

  if (fabs(mLastInsTimestamp) < 1e-4) {
    mLastInsTimestamp = rtk->timestamp / 1000000.0;
  }

  int priority = -1;
  if (match_config.priority == mLastInsPriority) {
    // status is not changed
    priority = match_config.priority;
    mLastInsTimestamp = rtk->timestamp / 1000000.0;
  } else if (match_config.priority < mLastInsPriority) {
    // status downgrade immediately
    LOG_WARN("SLAM: rtk status downgrade from {} to {}",
      mInsConfig[mLastInsPriority].status_name,
      (match_config.priority == -1 ? "invalid" : mInsConfig[match_config.priority].status_name)
    );
    priority = match_config.priority;
    mLastInsPriority = match_config.priority;
    mLastInsTimestamp = rtk->timestamp / 1000000.0;
  } else if (match_config.priority > mLastInsPriority) {
    double keep_time = rtk->timestamp / 1000000.0 - mLastInsTimestamp;
    if (keep_time >= mInsConfig[match_config.priority].stable_time) {
      LOG_INFO("SLAM: rtk status upgrade from {} to {}",
        (mLastInsPriority == -1 ? "invalid" : mInsConfig[mLastInsPriority].status_name),
        mInsConfig[match_config.priority].status_name
      );
      priority = match_config.priority;
      mLastInsPriority = match_config.priority;
      mLastInsTimestamp = rtk->timestamp / 1000000.0;
    } else {
      LOG_INFO("SLAM: rtk status is upgrading from {} to {}",
        (mLastInsPriority == -1 ? "invalid" : mInsConfig[mLastInsPriority].status_name),
        mInsConfig[match_config.priority].status_name
      );
      priority = std::max(mLastInsPriority, match_config.priority - 2);
    }
  }

  if (priority < 0) {
    return false;
  }

  rtk->dimension = mInsDim;
  rtk->precision = mInsConfig[priority].precision;
  return true;
}

bool SLAM::run(const uint64_t &timestamp,
               std::map<std::string, PointCloudAttrPtr>& points,
               std::map<std::string, ImageType> &images,
               std::map<std::string, cv::Mat> &images_stream,
               RTKType &rtk, std::vector<ImuType> &imu, PoseType &pose) {
  std::shared_ptr<RTKType> rtk_ptr = std::make_shared<RTKType>(rtk);

  // pre-process rtk data
  bool rtk_valid = preprocessInsData(rtk_ptr);

  // process GPS data
  if (mUseGPS && rtk_valid) {
    if (!mSlam->originIsSet()) {
      mSlam->setOrigin(rtk);
    }
    mSlam->feedInsData(rtk_ptr);
  }

  // process IMU data
  if (mUseIMU) {
    for (auto &imu_data : imu) {
      mSlam->feedImuData(imu_data);
    }
  }

  // process images
  mSlam->feedImageData(timestamp, images, images_stream);

  // process pointcloud
  mSlam->feedPointData(timestamp, points);

  // get odometry
  PointCloudAttrImagePose frame;
  Eigen::Matrix4d odom = mSlam->getPose(frame);

  // enqueue mapping
  if (mMappingMode != modeType::Localization) {
    mMapQueue.enqueue(frame);
  }

  // mSlam->getTimedPose(rtk_ptr->timestamp, odom);
  pose.T         = odom;
  pose.timestamp = frame.points->cloud->header.stamp;

  // localization mode
  if (mMappingMode == modeType::Localization) {
    pose.status = rtk.status;
    // state
    if (mSlam->isInited()) {
      if (!rtk_valid || !mUseGPS) {
        pose.state = "Localizing(L)";
      } else {
        pose.state = "Localizing(L+G)";
      }
    } else {
      pose.state   = "Initializing";
    }

    // lat, lon
    if (mSlam->isInited() && mZeroUtm) {
      mProjector->FromLocalToGlobal((*mZeroUtm)(0) + odom(0, 3), (*mZeroUtm)(1) + odom(1, 3), pose.latitude, pose.longitude);
      pose.altitude = (*mZeroUtm)(2) + odom(2, 3);
    } else {
      pose.latitude  = 0;
      pose.longitude = 0;
      pose.altitude  = 0;
    }
  } else {
    pose.latitude  = rtk.latitude;
    pose.longitude = rtk.longitude;
    pose.altitude  = rtk.altitude;
    pose.status    = rtk.status;
    pose.state     = "Mapping";
  }

  return true;
}

RTKType &SLAM::getOrigin() {
  return mSlam->getOrigin();
}

void SLAM::setOrigin(RTKType &rtk) {
  if (!mSlam->originIsSet()) {
    mSlam->setOrigin(rtk);
  }
}

void SLAM::mergeMap(const std::string& directory, std::vector<std::shared_ptr<KeyFrame>> &frames) {
  Locate::Localization *locater = dynamic_cast<Locate::Localization *>(mSlam.get());
  if (locater != nullptr) {
    locater->mergeMap(directory, frames);
  }
}

void SLAM::runMappingThread() {
  prctl(PR_SET_NAME, "Mapping", 0, 0, 0);

  while (mThreadStart) {
    PointCloudAttrImagePose frame;
    if (false == mMapQueue.wait_dequeue_timed(frame, 100000)) {
      continue;
    }

    PointCloudAttrImagePose keyframe;
    bool is_key_frame = enqueue_graph(frame, keyframe);
    if (is_key_frame) {
      // outlier remove
      PointCloud::Ptr filtered(new PointCloud());
      mOutlierRemoval->setInputCloud(keyframe.points->cloud);
      mOutlierRemoval->filter(*filtered);
      filtered->header = keyframe.points->cloud->header;

      // distance filter
      PointCloud::Ptr dist_filtered(new PointCloud());
      pointsDistanceFilter(filtered, dist_filtered, 0, mInitParameter.key_frame_range);

      keyframe.points->cloud = dist_filtered;
      keyframe.images = std::map<std::string, ImageType>(); // release I420 image for saving memory
      // enqueue keyframe queue
      mKeyframeQueue.enqueue(keyframe);
    }
  }
}

void SLAM::runLocalizationThread() {
  prctl(PR_SET_NAME, "SLAM Output", 0, 0, 0);
  LOG_INFO("start localization output, destination {}:{}", mDestinationIP, mDestinationPort);

  bool isLocalizationOutput = false;
  bool isImuOnline = true;
  std::unique_ptr<UDPServer> slamUDPServer(new UDPServer(0));
  while (mThreadStart) {
    std::string message;
    isImuOnline = (mUnixServer->Recv(message, isImuOnline ? 100000 : 10000) != 0);

    RTKType rtk;
    bool found = parseGPCHC(message, rtk);
    if (found || (mRunMode == runType::Online && !isImuOnline)) {
      Eigen::Matrix4d pose;
      bool pose_valid = false;
      if (isImuOnline) {
        pose_valid = mSlam->getTimedPose(rtk, pose);
      } else {
        rtk.timestamp = getCurrentTime();
        pose_valid = mSlam->getTimedPose(rtk.timestamp, pose);
        LOG_DEBUG("localization has no imu to predict, only based on motion model");
      }

      if (!pose_valid && !isImuOnline) {
        continue;
      } else if (!pose_valid && isImuOnline) {
        if (isImuOnline && isLocalizationOutput) {
          isLocalizationOutput = false;
          LOG_WARN("localization output switch to rtk only mode");
        }
      } else {
        if (isImuOnline && !isLocalizationOutput) {
          isLocalizationOutput = true;
          LOG_INFO("localization output switch to fusion mode");
        }
        if (mZeroUtm) {
          mProjector->FromLocalToGlobal((*mZeroUtm)(0) + pose(0, 3), (*mZeroUtm)(1) + pose(1, 3), rtk.latitude, rtk.longitude);
          rtk.altitude = (*mZeroUtm)(2) + pose(2, 3);
        } else {
          rtk.latitude  = 0;
          rtk.longitude = 0;
          rtk.altitude  = 0;
        }

        // publish fusion pose message
        nav_msgs::Odometry odometry_message;
        odometry_message.header.stamp = rtk.timestamp;
        fromOdometry(pose, odometry_message);
        get_core()->publish("slam.odometry", &odometry_message);

        double x, y, z;
        double yaw, pitch, roll;
        getRPYTfromTransformFrom(pose, x, y, z, yaw, pitch, roll);
        if (fabs(roll) >= 90.0 || fabs(pitch) >= 90.0) {
          pose = getTransformFromRPYT(x, y, z, -yaw, pitch, roll);
          getRPYTfromTransformFrom(pose, x, y, z, yaw, pitch, roll);
        } else {
          yaw = -yaw;
        }
        // heading range 0 ~ 360
        if (yaw < 0) {
          yaw = yaw + 360;
        }
        rtk.heading = yaw;
        rtk.pitch = pitch;
        rtk.roll = roll;
        message = formatGPCHC(rtk);
      }

      // publish localization message
      if (get_core_enable()) {
        sensor_msgs::NavSatFix nav_messgae;
        nav_messgae.header.stamp = rtk.timestamp;
        nav_messgae.latitude = rtk.latitude;
        nav_messgae.longitude = rtk.longitude;
        nav_messgae.altitude = rtk.altitude;
        PUBLISH_MSG("slam.nav", nav_messgae);
      }

      if (mOutput) {
        slamUDPServer->UDPSendto(mDestinationIP, mDestinationPort, message, message.length());
      }
    }
  }
}