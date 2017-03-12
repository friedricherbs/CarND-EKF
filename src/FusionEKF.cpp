#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
              0,      0.0225;

  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0,      0,
              0,    0.0009, 0,
              0,    0,      0.09;

  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1,0,0,0,
              0,1,0,0;
  Hj_ = MatrixXd(3, 4);

  //create a 4D state vector, we don't know yet the values of the x state
  ekf_.x_ = VectorXd(4);

  //state covariance matrix P
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0,    0,
	         0, 1, 0,    0,
	         0, 0, 1000, 0,
	         0, 0, 0,    1000;


  //measurement covariance
  ekf_.R_ = MatrixXd(2, 2);
  ekf_.R_ << 0.0225, 0,
	         0, 0.0225;

  //measurement matrix
  ekf_.H_ = MatrixXd(2, 4);
  ekf_.H_ << 1, 0, 0, 0,
	         0, 1, 0, 0;

  //the initial transition matrix F_
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
	         0, 1, 0, 1,
	         0, 0, 1, 0,
	         0, 0, 0, 1;

  //set the acceleration noise components
  noise_ax = 9.0; //0.005; 
  noise_ay = 9.0; //0.008;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    //cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      const double cos_theta = cos(measurement_pack.raw_measurements_[1]);
      const double sin_theta = sin(measurement_pack.raw_measurements_[1]);
      const double x_init    = measurement_pack.raw_measurements_[0] * cos_theta;
      const double y_init    = measurement_pack.raw_measurements_[0] * sin_theta;
      const double vx_init   = measurement_pack.raw_measurements_[2] * cos_theta;
      const double vy_init   = measurement_pack.raw_measurements_[2] * sin_theta;

      ekf_.x_ = VectorXd(4);
      ekf_.x_ << x_init, y_init, vx_init, vy_init;

      //set the initial convariance matrix
      Hj_ = tools.CalculateJacobian(ekf_.x_);
      ekf_.H_ = MatrixXd(3, 4); 
      ekf_.H_ = Hj_;

      ekf_.R_ = MatrixXd(3, 3);  
      ekf_.R_ = R_radar_;

      // Calculate initial covariance matrix via error propagation

      // Derivative of state coordinates with respect to ro
      const double dxdro  = cos_theta;
      const double dydro  = sin_theta;
      const double dvxdro = cos_theta;
      const double dvydro = sin_theta;

      // Derivative of state coordinates with respect to theta
      const double dxdtheta  = -measurement_pack.raw_measurements_[0] * sin_theta;
      const double dydtheta  = measurement_pack.raw_measurements_[0]  * cos_theta;
      const double dvxdtheta = -measurement_pack.raw_measurements_[2] * sin_theta;
      const double dvydtheta = measurement_pack.raw_measurements_[2]  * cos_theta;

      // Derivative of state coordinates with respect to ro dot
      const double dxdrodot  = 0;
      const double dydrodot  = 0;
      const double dvxdrodot = cos_theta;
      const double dvydrodot = sin_theta;

      // Do error propagation
      const double var_x  = dxdro*dxdro*R_radar_(0,0)   + dxdtheta*dxdtheta*R_radar_(1,1)   + dxdrodot*dxdrodot*R_radar_(2,2);
      const double var_y  = dydro*dydro*R_radar_(0,0)   + dydtheta*dydtheta*R_radar_(1,1)   + dydrodot*dydrodot*R_radar_(2,2);
      const double var_vx = dvxdro*dvxdro*R_radar_(0,0) + dvxdtheta*dvxdtheta*R_radar_(1,1) + dvxdrodot*dvxdrodot*R_radar_(2,2);
      const double var_vy = dvydro*dvydro*R_radar_(0,0) + dvydtheta*dvydtheta*R_radar_(1,1) + dvydrodot*dvydrodot*R_radar_(2,2);

      ekf_.P_ << var_x,0,0,0,
                 0,    var_y,0,0,
                 0,0,var_vx,0,
                 0,0,0,var_vy; 
    }
	else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
		/**
		Initialize state.
		*/
		//set the state with the initial location and zero velocity
		ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;

        //set the initial convariance matrix
        ekf_.P_ << R_laser_(0,0), 0,             0,    0,
                   0,             R_laser_(1,1), 0,    0,
                   0,             0,             1000, 0,
                   0,             0,             0,    1000;
	}

	previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
   */

  //compute the time elapsed between the current and previous measurements
  const double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  const double dt_2 = dt * dt;
  const double dt_3 = dt_2 * dt;
  const double dt_4 = dt_3 * dt;

  //Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  //set the process covariance matrix Q
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ <<  dt_4/4*noise_ax,    0,                dt_3/2*noise_ax, 0,
	          0,                  dt_4/4*noise_ay,  0, dt_3/2*noise_ay,
	          dt_3/2*noise_ax,    0,                dt_2*noise_ax, 0,
	          0, dt_3/2*noise_ay, 0,                dt_2*noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
      Hj_ = tools.CalculateJacobian(ekf_.x_);
      ekf_.H_ = MatrixXd(3, 4); 
      ekf_.H_ = Hj_;

      ekf_.R_ = MatrixXd(3, 3);  
      ekf_.R_ = R_radar_;

      ekf_.UpdateEKF(measurement_pack.raw_measurements_, &radar_meas_fct);
  } else {
      ekf_.R_ = MatrixXd(2, 2);
      ekf_.R_ = R_laser_;

      ekf_.H_ = MatrixXd(2, 4);
      ekf_.H_ = H_laser_;

      //ekf_.Update(measurement_pack.raw_measurements_);
  }
}

bool FusionEKF::radar_meas_fct(const VectorXd& state, VectorXd& measurements)
{
    bool ret_val           = false;
    const double ro_state  = sqrt(state(0)*state(0)+state(1)*state(1));
    if (abs(ro_state) > 0.0001)
    {
        ret_val = true;
        const double theta_state  = atan2(state(1),state(0));
        const double ro_dot_state = (state(0)*state(2)+state(1)*state(3))/ro_state;

        measurements << ro_state, theta_state, ro_dot_state;
    }
    return ret_val;
}
