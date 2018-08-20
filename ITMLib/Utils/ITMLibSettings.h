// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include <memory>
#include "../Objects/ITMSceneParams.h"
#include "../Engine/ITMTracker.h"
#include "../Engine/ITMGroundTruthTracker.h"
#include "VoxelDecayParams.h"

namespace ITMLib
{
	namespace Objects
	{
		class ITMLibSettings
		{
		public:
			/// The device used to run the DeviceAgnostic code
			typedef enum {
				DEVICE_CPU,
				DEVICE_CUDA,
				DEVICE_METAL
			} DeviceType;

			/// Select the type of device to use
            DeviceType deviceType;

            InfiniTAM::VoxelDecayParams decayParams;
			/// Enables swapping between host and device.
			bool useSwapping;

			bool useApproximateRaycast;

			bool useBilateralFilter;

			bool modelSensorNoise;

			/// Tracker types
			typedef enum {
				//! Identifies a tracker based on colour image
				TRACKER_COLOR,
				//! Identifies a tracker based on depth image
				TRACKER_ICP,
				//! Identifies a tracker based on depth image (Ren et al, 2012)
				TRACKER_REN,
				//! Identifies a tracker based on depth image and IMU measurement
				TRACKER_IMU,
				//! Identifies a tracker that use weighted ICP only on depth image
				TRACKER_WICP,
				//! Identifies a tracker which uses a set of known poses.
				TRACKER_GROUND_TRUTH,
				//! Identifies a NOP tracker symbolizing that the ITM pose should be set directly.
				TRACKER_EXTERNAL
			} TrackerType;

			/// Select the type of tracker to use
			TrackerType trackerType;

			/// The tracking regime used by the tracking controller
			/// TODO(andrei): Handle this correctly on object copy.
			TrackerIterationType *trackingRegime;

			/// The number of levels in the trackingRegime
			int noHierarchyLevels;
			
			/// Run ICP till # Hierarchy level, then switch to ITMRenTracker for local refinement.
			int noICPRunTillLevel;

			/// For ITMColorTracker: skip every other point in energy function evaluation.
			bool skipPoints;

			/// For ITMDepthTracker: ICP distance threshold
			float depthTrackerICPThreshold;

			/// For ITMDepthTracker: ICP iteration termination threshold
			float depthTrackerTerminationThreshold;

			/// Further, scene specific parameters such as voxel size
			ITMLib::Objects::ITMSceneParams sceneParams;

			// For ITMGroundTruthTracker: The location of the ground truth pose
			// information folder. Currently, only the OxTS format is supported (the
			// one in which the KITTI dataset ground truth pose information is
			// provided).
			std::string groundTruthPoseFpath;

			// Used for the instance reconstruction in conjunction with the ground truth
			// tracker. Enables the object-specific InfiniTAM to start "tracking" from the
			// correct frame.
			int groundTruthPoseOffset;

			/// \brief The number of voxel blocks stored on the GPU.
			/// This imposes a hard limit on the maximum
			long sdfLocalBlockNum;

			// Whether to create all the things required for marching cubes and mesh extraction.
			// - uses additional memory (lots!)
			bool createMeshingEngine = true;

			// maxW gets set to this when dynamic fusion weights (which depend on the depth of each
			// measurement) are enabled.
			static const int maxWDynamic = 50000;

			ITMLibSettings();
			ITMLibSettings(const ITMLibSettings &other) = default;
			~ITMLibSettings();

			// Suppress the default copy constructor and assignment operator
			// Re-enabled for DynSLAM, since we need to pass modified copies of the settings object
			// from the static environment reconstructor to the object instance reconstructors.
//			ITMLibSettings(const ITMLibSettings&);
//			ITMLibSettings& operator=(const ITMLibSettings&);
		};
	}
}
