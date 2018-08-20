
#ifndef INFINITAM_ITMGROUNDTRUTHTRACKER_H
#define INFINITAM_ITMGROUNDTRUTHTRACKER_H

#include <iostream>
#include <fstream>

#include "ITMTracker.h"
#include "../Utils/ITMOxtsIO.h"

#define KITTI 0
#define DADAO 1

namespace ITMLib {
  namespace Engine {

	using namespace ITMLib::Objects;
	using namespace std;

	/**
	 * Dummy tracker which relays pose information from a file.
	 * The currently supported file format is the ground truth odometry information from
	 * the KITTI odometry dataset.
	 *
	 * Note that this information is not a 100% "cheaty" ground truth computed using, e.g.,
	 * manual annotation or beacons, but the one recorded by the vehicle's IMG/GPS module.
	 *
	 * In the future, nevertheless, it would be much cooler to just do this using sparse-to-dense
	 * odometry from stereo (and maybe lidar) data.
	 */
	class ITMGroundTruthTracker : public ITMTracker {

	private:
		int currentFrame;
		vector<Matrix4f> groundTruthPoses;
		Matrix4f currentPose;

		Matrix4f readPose(istream &in) {
			Matrix4f pose;
			in >> pose.m00 >> pose.m10 >> pose.m20 >> pose.m30
				>> pose.m01 >> pose.m11 >> pose.m21 >> pose.m31
				>> pose.m02 >> pose.m12 >> pose.m22 >> pose.m32;
			pose.m03 = pose.m13 = pose.m23 = 0.0f;
			pose.m33 = 1.0f;

			return pose;
		}

        Matrix4f readDadaoPose(istream &in) {
            Matrix4f pose;
            int num,time;
            float x,y,theta;
            in >> num >> time >> x >> y >> theta;
            theta = -theta;
            pose.m00 = cos(theta);
            pose.m10 = 0;
            pose.m20 = sin(theta);
            pose.m30 = y;
            pose.m01 = 0;
            pose.m11 = 1;
            pose.m21 = 0;
            pose.m31 = 0;
            pose.m02 = -sin(theta);
            pose.m12 = 0;
            pose.m22 = cos(theta);
            pose.m32 = x;
            pose.m03 = pose.m13 = pose.m23 = 0.0f;
            pose.m33 = 1.0f;

            return pose;
        }

	  // TODO(andrei): Move this helper out of here.
		/// \brief Loads a KITTI-odometry ground truth pose file.
		/// \return A list of relative vehicle poses expressed as 4x4 matrices.
      vector<Matrix4f> readKittiOdometryPoses(const std::string &fpath) {

        ifstream fin(fpath);
		  if (! fin.is_open()) {
		  throw runtime_error("Could not open odometry file.");
		}

		cout << "Loading odometry ground truth from file: " << fpath << endl;

		// Keep track of relative transforms, which makes it easier to start reconstructing from an
		// arbitrary frame, among other things.
		vector<Matrix4f> poses;
		Matrix4f first = readPose(fin);
		Matrix4f inv_prev;
		first.inv(inv_prev);
		Matrix4f eye;
		eye.setIdentity();
		poses.push_back(eye);

		while (! fin.eof()) {
			// This matrix takes a point in the ith coordinate system, and projects it
			// into the first (=0th, or world) coordinate system.
			//
			// The M matrix in InfiniTAM is a modelview matrix, so it transforms points from
			// the world coordinates, to camera coordinates.
			//
			// One therefore needs to set this pose as 'InvM' in the InfiniTAM tracker state.
			Matrix4f new_pose = readPose(fin);
			Matrix4f rel_pose = inv_prev * new_pose;
			new_pose.inv(inv_prev);
			poses.push_back(rel_pose);
		}

		return poses;
	  }

      // TODO(andrei): Move this helper out of here.
        /// \brief Loads a Dadao-odometry ground truth pose file.
        /// \return A list of relative vehicle poses expressed as 4x4 matrices.
      vector<Matrix4f> readDadaoOdometryPoses(const std::string &fpath) {

        ifstream fin(fpath);
          if (! fin.is_open()) {
          throw runtime_error("Could not open odometry file.");
        }

        cout << "Loading odometry ground truth from file: " << fpath << endl;

        // Keep track of relative transforms, which makes it easier to start reconstructing from an
        // arbitrary frame, among other things.
        vector<Matrix4f> poses;
        Matrix4f first = readDadaoPose(fin);
        Matrix4f inv_prev;
        first.inv(inv_prev);
        Matrix4f eye;
        eye.setIdentity();
        poses.push_back(eye);

        while (! fin.eof()) {
            // This matrix takes a point in the ith coordinate system, and projects it
            // into the first (=0th, or world) coordinate system.
            //
            // The M matrix in InfiniTAM is a modelview matrix, so it transforms points from
            // the world coordinates, to camera coordinates.
            //
            // One therefore needs to set this pose as 'InvM' in the InfiniTAM tracker state.
            Matrix4f new_pose = readDadaoPose(fin);
            Matrix4f rel_pose = inv_prev * new_pose;
            new_pose.inv(inv_prev);
            poses.push_back(rel_pose);
        }

        return poses;
      }

		// Taken from 'ITMPose' to aid with debugging.
		// TODO(andrei): Move to common utility. Can make math-intense code much cleaner.
		Matrix3f getRot(const Matrix4f M) {
			Matrix3f R;
			R.m[0 + 3*0] = M.m[0 + 4*0]; R.m[1 + 3*0] = M.m[1 + 4*0]; R.m[2 + 3*0] = M.m[2 + 4*0];
			R.m[0 + 3*1] = M.m[0 + 4*1]; R.m[1 + 3*1] = M.m[1 + 4*1]; R.m[2 + 3*1] = M.m[2 + 4*1];
			R.m[0 + 3*2] = M.m[0 + 4*2]; R.m[1 + 3*2] = M.m[1 + 4*2]; R.m[2 + 3*2] = M.m[2 + 4*2];
			return R;
		}

	protected:

	public:
		ITMGroundTruthTracker(const string &groundTruthFpath, int frameOffset) {

            if( this->groundTruthMode == KITTI ){
                cout << "Created Kitti ground truth-based tracker. Will read data from: "
                     << groundTruthFpath << " with offset " << frameOffset << "." << endl;
                groundTruthPoses = readKittiOdometryPoses(groundTruthFpath);
            } else if ( this->groundTruthMode == DADAO ){
                cout << "Created Dadao ground truth-based tracker. Will read data from: "
                     << groundTruthFpath << " with offset " << frameOffset << "." << endl;
                groundTruthPoses = readDadaoOdometryPoses(groundTruthFpath);
            }
			this->currentFrame = frameOffset;
			this->currentPose.setIdentity();

			// TODO(andrei): This code, although untested, should provide a skeleton for
			// reading OxTS data, such that ground truth from the full KITTI dataset can
			// be read.
//        vector<OxTSFrame> groundTruthFrames = Objects::readOxtsliteData(groundTruthFpath);
//        groundTruthPoses = Objects::oxtsToPoses(groundTruthFrames, groundTruthTrans, groundTruthRots);
		}

		void TrackCamera(ITMTrackingState *trackingState, const ITMView *view) {
			// Start with this, since tracking never happens for the first frame (since there's
			// nothing to track).
			this->currentFrame++;
			Matrix4f Minc = groundTruthPoses[currentFrame];
			currentPose = currentPose * Minc;
			cout << "Current pose in GT tracker:" << endl << currentPose << endl;
			trackingState->pose_d->SetInvM(currentPose);
		}

	  // Note: this doesn't seem to get used much in InfiniTAM. It's just
	  // called from 'ITMMainEngine', but none of its implementations
	  // currently do anything.
	  void UpdateInitialPose(ITMTrackingState *trackingState) {}

      //groundtruth mode , contains Dadao dataset and Kitti dataset.
      static int groundTruthMode;

	  virtual ~ITMGroundTruthTracker() {}

	};
  }
}

#endif //INFINITAM_ITMGROUNDTRUTHTRACKER_H
