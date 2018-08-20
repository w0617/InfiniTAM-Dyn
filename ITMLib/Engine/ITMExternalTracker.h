#ifndef DYNSLAM_ITMEXTERNALTRACKER_H
#define DYNSLAM_ITMEXTERNALTRACKER_H

#include "ITMTracker.h"

namespace ITMLib {
namespace Engine {

class ITMExternalTracker : public ITMTracker {
 public:
  ITMExternalTracker() = default;

  void TrackCamera(ITMTrackingState *trackingState, const ITMView *view) override;
};

}
}

#endif //DYNSLAM_ITMEXTERNALTRACKER_H
