#include <iostream>
#include "tools.h"

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
    const vector<VectorXd> &ground_truth) 
{
    VectorXd rmse(4);
    rmse << 0,0,0,0;

    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size
    if(estimations.size() != ground_truth.size()
        || estimations.size() == 0){
            cout << "Invalid estimation or ground_truth data" << endl;
            return rmse;
    }

    //accumulate squared residuals
    for(unsigned int i=0; i < estimations.size(); ++i)
    {

        VectorXd residual = estimations[i] - ground_truth[i];

        //coefficient-wise multiplication
        residual = residual.array()*residual.array();
        rmse += residual;
    }

    //calculate the mean
    rmse = rmse/estimations.size();

    //calculate the squared root
    rmse = rmse.array().sqrt();

    //return the result
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

	MatrixXd Hj(3,4);
	//recover state parameters
	const double px = x_state(0);
	const double py = x_state(1);
	const double vx = x_state(2);
	const double vy = x_state(3);

	//pre-compute a set of terms to avoid repeated calculation
	const double c1 = px*px+py*py;
	const double c2 = sqrt(c1);
	const double c3 = (c1*c2);

	//check division by zero
	if(fabs(c1) < 0.0001){
		cout << "CalculateJacobian () - Error - Division by Zero" << endl;
		return Hj;
	}

	//compute the Jacobian matrix
	Hj << (px/c2),               (py/c2),               0,     0,
	   	 -(py/c1),               (px/c1),               0,     0,
		  py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

	return Hj;
}

void Tools::EstimateMeasurementNoise(const vector<MeasurementPackage>& measurements, const vector<GroundTruthPackage>& ground_truth)
{
    const size_t N = measurements.size();
    if (ground_truth.size() != N || ground_truth.size() == 0)
    {
        cout << "Size error. Cannot proceed!" << endl;
        return;
    }

    // Calculate variances via Welford's method
    double m_x_laser = 0.0;
    double s_x_laser = 0.0;
    double m_y_laser = 0.0;
    double s_y_laser = 0.0;

    double m_ro_radar      = 0.0;
    double s_rho_radar     = 0.0;
    double m_theta_radar   = 0.0;
    double s_theta_radar   = 0.0;
    double m_ro_dot_radar  = 0.0;
    double s_ro_dot_radar = 0.0;
    
    long N_laser           = 0;
    long N_radar           = 0;

    for (size_t k = 0; k < N; ++k) {
        const double x_gt  =  ground_truth[k].gt_values_(0);
        const double y_gt  =  ground_truth[k].gt_values_(1);
        const double vx_gt =  ground_truth[k].gt_values_(2);
        const double vy_gt =  ground_truth[k].gt_values_(3);

        if (measurements[k].sensor_type_ == MeasurementPackage::LASER)
        {
            const double x_meas       = measurements[k].raw_measurements_(0);
            const double y_meas       = measurements[k].raw_measurements_(1); 

            const double innovation_x = x_gt - x_meas;
            const double innovation_y = y_gt - y_meas;

            const double oldm_x_laser = m_x_laser;
            const double oldm_y_laser = m_y_laser;

            m_x_laser                += (innovation_x-m_x_laser)/(k+1);
            s_x_laser                += (innovation_x-m_x_laser)*(innovation_x-oldm_x_laser);

            m_y_laser                += (innovation_y-m_y_laser)/(k+1);
            s_y_laser                += (innovation_y-m_y_laser)*(innovation_y-oldm_y_laser);

            ++N_laser;
        }
        else if (measurements[k].sensor_type_ == MeasurementPackage::RADAR) {
            // output the estimation in the cartesian coordinates
            const double ro_gt     = sqrt(x_gt*x_gt+y_gt*y_gt);
            const double theta_gt  = atan2(y_gt,x_gt);
            const double ro_dot_gt = (x_gt*vx_gt+y_gt*vy_gt)/ro_gt;

            const double ro_meas      = measurements[k].raw_measurements_(0);
            const double theta_meas   = measurements[k].raw_measurements_(1); 
            const double ro_dot_meas  = measurements[k].raw_measurements_(2);

            const double innovation_ro     = ro_gt - ro_meas;
            const double innovation_theta  = theta_gt - theta_meas;
            const double innovation_ro_dot = ro_dot_gt - ro_dot_meas;

            const double oldm_ro_radar = m_ro_radar;
            const double oldm_theta_radar = m_theta_radar;
            const double oldm_ro_dot_radar = m_ro_dot_radar;

            m_ro_radar     += (innovation_ro-m_ro_radar)/(k+1);
            s_rho_radar    += (innovation_ro-m_ro_radar)*(innovation_ro-oldm_ro_radar);
            m_theta_radar  += (innovation_theta-m_theta_radar)/(k+1);
            s_theta_radar  += (innovation_theta-m_theta_radar)*(innovation_theta-oldm_theta_radar);
            m_ro_dot_radar += (innovation_ro_dot-m_ro_dot_radar)/(k+1);
            s_ro_dot_radar += (innovation_ro_dot-m_ro_dot_radar)*(innovation_ro_dot-oldm_ro_dot_radar);

            ++N_radar;
        }
        else
        {
            continue;
        }
    }

    if (N_laser > 1)
    {
        const double var_x_laser = s_x_laser/(N_laser-1);
        const double var_y_laser = s_y_laser/(N_laser-1);
        cout << "VarX Laser: " << var_x_laser << "VarY Laser: " << var_y_laser << endl;
    }

    if (N_radar > 1)
    {
        const double var_rho_radar     = s_rho_radar/(N_radar-1);
        const double var_theta_radar   = s_theta_radar/(N_radar-1);
        const double var_ro_dot_radar = s_ro_dot_radar/(N_radar-1);
        cout << "Var_Ro Radar: " << var_rho_radar << "Var_Theta Radar: " << var_theta_radar << "Var_Ro_Dot Radar: " << var_ro_dot_radar << endl;
    }
}

void Tools::EstimateProcessNoise(const vector<GroundTruthPackage>& ground_truth)
{
  const size_t N  = ground_truth.size();
  double m_ax     = 0.0;
  double s_ax     = 0.0;
  double m_ay     = 0.0;
  double s_ay     = 0.0;

  long long num_gt = 0;

  for (size_t k = 1; k < N; ++k) {
      const double vx_gt     =  ground_truth[k].gt_values_(2);
      const double vy_gt     =  ground_truth[k].gt_values_(3);
      const long long t1     =  ground_truth[k].timestamp_;

      const double vx_old_gt =  ground_truth[k-1].gt_values_(2);
      const double vy_old_gt =  ground_truth[k-1].gt_values_(3);
      const long long t0     =  ground_truth[k-1].timestamp_;

      const double dt        = (t1-t0)/ 1000000.0;

      if (dt > 0.00001)
      {
        const double ax = (vx_gt - vx_old_gt)/dt;
        const double ay = (vy_gt - vy_old_gt)/dt;

        const double m_ax_old = m_ax;
        m_ax += (ax-m_ax)/k;
        s_ax += (ax-m_ax)*(ax-m_ax_old);

        const double m_ay_old = m_ay;
        m_ay += (ay-m_ay)/k;
        s_ay += (ay-m_ay)*(ay-m_ay_old);

        ++num_gt;
      }    
  }

  if(num_gt > 1)
  {
      const double var_ax = s_ax/(num_gt-1);
      const double var_ay = s_ay/(num_gt-1);

      cout << "Q for AX: " << var_ax << "Q for AY: " << var_ay << endl;
  }

}
