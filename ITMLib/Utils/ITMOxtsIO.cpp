
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "ITMOxtsIO.h"
#include "../Utils.h"

using namespace std;

namespace ITMLib {
  // TODO(andrei): Better namespace name?
  namespace Objects {

    void readTimestampWithNanoseconds(
        const string &input,
        tm *time,
        long *nanosecond
    ) {
      int year, month, day, hour, minute, second;
      sscanf(input.c_str(), "%d-%d-%d %d:%d:%d.%ld", &year, &month, &day, &hour,
             &minute, &second, nanosecond);
      time->tm_year = year;
      time->tm_mon = month - 1;
      time->tm_mday = day;
      time->tm_hour = hour;
      time->tm_min = minute;
      time->tm_sec = second;
    }

    vector<tm> readTimestamps(const string& dir) {
      const string timestampFpath = dir + "/timestamps.txt";
      ifstream fin(timestampFpath.c_str());
      vector<tm> timestamps;

      string line;
      while(getline(fin, line)) {
        tm timestamp;
        long ns;     // We ignore this for now.
        readTimestampWithNanoseconds(line, &timestamp, &ns);
        timestamps.push_back(timestamp);
      }

      return timestamps;
    }

    /// \brief Customized matrix printing useful for debugging.
    void prettyPrint(ostream &out, const Matrix4f& m) {
	    out << "[";
      for (size_t row = 0; row < 4; ++row) {
	      if (row > 0) {
		      out << " ";
	      }
	      out << "[";
        for (size_t col = 0; col < 4; ++col) {
          out << setw(6) << setprecision(4) << fixed << right << m.m[col * 4 + row];
	        if (col < 3) {
		        out << ", ";
	        }
        }
        out << "]" << endl;
      }
    }

    /// \brief Compute the Mercator scale from the latitude.
    double latToScale(double latitude) {
      return cos(latitude * M_PI / 180.0);
    }

    /// \brief Converts lat/lon coordinates to Mercator coordinates.
    Vector2d latLonToMercator(double latitude, double longitude, double scale) {
      const double EarthRadius = 6378137.0;
      double mx = scale * longitude * M_PI * EarthRadius / 180.0;
      double my = scale * EarthRadius * log(tan( (90 + latitude) * M_PI / 360 ));
      return Vector2d(mx, my);
    }

    OxTSFrame readOxtslite(const string& fpath) {
      ifstream fin(fpath.c_str());
      if (! fin.is_open()) {
        throw runtime_error(dynslam::utils::Format("Could not open pose file [%s].", fpath.c_str()));
      }
      if (fin.bad()) {
        throw runtime_error(dynslam::utils::Format("Could not read pose file [%s].", fpath.c_str()));
      }

      OxTSFrame resultFrame;
      fin >> resultFrame.lat >> resultFrame.lon >> resultFrame.alt
          >> resultFrame.roll >> resultFrame.pitch >> resultFrame.yaw
          >> resultFrame.vn >> resultFrame.ve >> resultFrame.vf
          >> resultFrame.vl >> resultFrame.vu >> resultFrame.ax
          >> resultFrame.ay >> resultFrame.az >> resultFrame.af
          >> resultFrame.al >> resultFrame.au >> resultFrame.wx
          >> resultFrame.wy >> resultFrame.wz >> resultFrame.wf
          >> resultFrame.wl >> resultFrame.wu >> resultFrame.posacc
          >> resultFrame.velacc >> resultFrame.navstat >> resultFrame.numsats
          >> resultFrame.posmode >> resultFrame.velmode >> resultFrame.orimode;
      return resultFrame;
    }

    vector<Matrix4f> oxtsToPoses(const vector<OxTSFrame>& oxtsFrames,
                                 vector<Vector3f>& trans,
                                 vector<Matrix3f>& rots
    ) {
      double scale = latToScale(oxtsFrames[0].lat);
      vector<Matrix4f> poses;

      // TODO(andrei): Ensure matrix initialization order is correct. The
      // Matrix classes used here are COLUMN major!!
      // TODO(andrei): Ensure precision is consistent.

      // TODO(andrei): Document this very clearly.
      bool tr_initialized = false;
      // Keeps track of the inverse transform of the initial pose.
      Matrix4f tr_0_inv;

      for (size_t frameIdx = 0; frameIdx < oxtsFrames.size(); ++frameIdx) {
        const OxTSFrame &frame = oxtsFrames[frameIdx];
        // Extract the 3D translation information
        Vector2d translation2d = latLonToMercator(frame.lat, frame.lon, scale);
        Vector3f translation(translation2d.x, translation2d.y, frame.alt);

        // Extract the 3D rotation information from yaw, pitch, and roll.
        // See the OxTS user manual, pg. 71, for more information.
        float rollf = (float) frame.roll;
        float pitchf = (float) frame.pitch;
        float yawf = (float) frame.yaw;

        if (frameIdx < 10) {
          cout << rollf << " " << pitchf << " " << yawf << endl;
        }

        Matrix3f rotX(          1,           0,            0,
                                0,  cos(rollf),  -sin(rollf),
                                0,  sin(rollf),   cos(rollf));
        Matrix3f rotY( cos(pitchf),          0,   sin(pitchf),
                                 0,          1,             0,
                      -sin(pitchf),          0,   cos(pitchf));
        Matrix3f rotZ(   cos(yawf), -sin(yawf),             0,
                         sin(yawf),  cos(yawf),             0,
                                 0,          0,             1);

        Matrix3f rot = rotZ * rotY * rotX;

//        rots.push_back(rot);
//        trans.push_back(translation);

        Matrix4f transform;
        // TODO utility for this (setTransform(Matrix4f&, const Matrix3f&, const Vector3f&)
        for (int x = 0; x < 3; ++x) {
          for (int y = 0; y < 3; ++y) {
            transform(y, x) = rot(x, y);
          }
        }

        transform(3, 0) = (float) translation[0];
        transform(3, 1) = (float) translation[1];
        transform(3, 2) = (float) translation[2];
        cout << "Translation: " << translation << endl;

        transform(0, 3) = 0;
        transform(1, 3) = 0;
        transform(2, 3) = 0;
        transform(3, 3) = 1;

//        cout << "Transform [" << frameIdx << "]:" << endl;
//        prettyPrint(cout, transform);

        // TODO make this more concise
        if (! tr_initialized) {
          cout << "Initializing inv transform to first frame." << endl;
          tr_initialized = true;
          if (!transform.inv(tr_0_inv)) {
            throw runtime_error(dynslam::utils::Format(
                "Ill-posed transform matrix inversion in frame %d.",
                frameIdx)
            );
          }
//          cout << "Here it is:" << endl << tr_0_inv << endl;
        }

        Matrix4f newPose = tr_0_inv * transform;
//        cout << "Pose [" << frameIdx << "]: " << endl;
//        prettyPrint(cout, newPose);
//        poses.push_back(transform);
        poses.push_back(newPose);

        if (frameIdx < 10) {
          cout << "New transform:" << endl << transform << endl;
          cout << "New transform (relative to first): " << endl << newPose
               << endl;
        }
      }

      // TODO(andrei): We may actually just require incremental poses, not
      // absolute ones. Documnent the output very clearly either way!

      return poses;
    }

    /// \brief Reads a set of ground truth OxTS IMU/GPU unit from a directory.
    /// \note This function, together with its related utilities, are based on
    /// the kitti devkit authored by Prof. Andreas Geiger.
    vector<OxTSFrame> readOxtsliteData(const string& dir) {
      // TODO(andrei): In the future, consider using the C++14 filesystem API
      // to make this code cleaner and more portable.
      vector<OxTSFrame> res;

      auto timestamps = readTimestamps(dir);
      cout << "Read " << timestamps.size() << " timestamps." << endl;

      for(size_t i = 0; i < timestamps.size(); ++i) {
        stringstream ss;
        ss << dir << "/data/" << setw(10) << setfill('0') << i << ".txt";
        cout << ss.str() << endl;

        // TODO(andrei): Should we expect missing data? Does that occur in
        // KITTI?
        res.push_back(readOxtslite(ss.str().c_str()));
      }

      return res;
    }

  }
}

