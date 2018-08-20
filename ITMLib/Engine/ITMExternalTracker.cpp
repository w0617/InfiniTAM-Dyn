#include "ITMExternalTracker.h"

void ITMLib::Engine::ITMExternalTracker::TrackCamera(ITMTrackingState *trackingState,
                                                     const ITMView *view) {
  throw std::runtime_error("The InfiniTAM engine should not attempt to initiate tracking when "
                           "using external tracking. The owner of the InfiniTAM engine should "
                           "instead use 'SetPose' to set the InfiniTAM pose accordingly.");
}
