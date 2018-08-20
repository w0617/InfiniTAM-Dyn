#pragma once

#include <cstdarg>
#include <iostream>
#include <vector>

#include <stdint.h>

#include "ITMMath.h"

namespace ITMLib {
  namespace Objects {

	  using namespace std;

	  /// \brief The IMU-GPU unit's state at a point in time.
	  struct OxTSFrame {
		  // latitude of the oxts-unit (deg)
		  double lat;
		  // longitude of the oxts-unit (deg)
		  double lon;
		  // altitude of the oxts-unit (m)
		  double alt;
		  //roll angle (rad),  0 = level, positive = left side up (-pi..pi)
		  double roll;
		  //itch angle (rad), 0 = level, positive = front down (-pi/2..pi/2)
		  double pitch;
		  // heading (rad),     0 = east,  positive = counter clockwise (-pi. .pi);
		  double yaw;
		  // velocity towards north (m/s)
		  double vn;
		  // velocity towards east (m/s);
		  double ve;
		  //  forward velocity, i.e. parallel to earth-surface (m/s)
		  double vf;
		  //  leftward velocity, i.e. parallel to earth-surface (m/s)
		  double vl;
		  //  upward velocity, i.e. perpendicular to earth-surface (m/s)
		  double vu;
		  //  acceleration in x, i.e. in direction of vehicle front (m/s^2)
		  double ax;
		  //  acceleration in y, i.e. in direction of vehicle left (m/s^2)
		  double ay;
		  //  acceleration in z, i.e. in direction of vehicle top (m/s^2)
		  double az;
		  //  forward acceleration (m/s^2)
		  double af;
		  //  leftward acceleration (m/s^2)
		  double al;
		  //  upward acceleration (m/s^2)
		  double au;
		  //  angular rate around x (rad/s)
		  double wx;
		  //  angular rate around y (rad/s)
		  double wy;
		  //  angular rate around z (rad/s)
		  double wz;
		  //  angular rate around forward axis (rad/s)
		  double wf;
		  //  angular rate around leftward axis (rad/s)
		  double wl;
		  //  angular rate around upward axis (rad/s)
		  double wu;
		  // velocity accuracy (north/east in m)
		  double posacc;
		  // velocity accuracy (north/east in m/s)
		  double velacc;
		  // navigation status
		  double navstat;
		  // number of satellites tracked by primary GPS receiver
		  double numsats;
		  // position mode of primary GPS receiver
		  double posmode;
		  // velocity mode of primary GPS receiver
		  double velmode;
		  // orientation mode of primary GPS receiver
		  double orimode;
	  };

	  /// Reads a full timestamp with nanosecond resolution seconds, such as
	  /// "2011-09-26 15:20:11.552379904".
	  /// Populates the standard C++ time object, plus an additional long containing
	  /// the nanoseconds (since the standard `tm` object only has second-level
	  /// accuracy).
	  ///
	  /// \param input Time string to parse.
	  /// \param time Second-resolution C++ time object (out parameter).
	  /// \param nanosecond Additional nanoseconds (out parameter).
	  void readTimestampWithNanoseconds(const std::string &input, tm *time,
	                                    long *nanosecond);

	  // \brief TODO doc
	  vector<tm> readTimestamps(const std::string &dir);

	  // \brief TODO doc
	  vector<OxTSFrame> readOxtsliteData(const std::string &dir);

	  vector<Matrix4f> oxtsToPoses(const vector<OxTSFrame> &oxtsFrames,
	                               vector<Vector3f> &trans,
	                               vector<Matrix3f> &rots);

	  void prettyPrint(ostream &out, const Matrix4f &m);
  }
}

