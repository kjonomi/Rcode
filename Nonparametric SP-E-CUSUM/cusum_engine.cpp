#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List update_online_cusum_cpp(double new_obs_z, NumericVector S_comp, 
                             NumericVector k_vals, NumericVector weights, double H) {
  int n_comp = k_vals.size();
  NumericVector new_S_comp(n_comp);
  double new_S_ens = 0.0;

  for(int j = 0; j < n_comp; j++) {
    double score = S_comp[j] + (new_obs_z - k_vals[j]);
    new_S_comp[j] = score > 0.0 ? score : 0.0;
    new_S_ens += new_S_comp[j] * weights[j];
  }

  bool alarm = (new_S_ens >= H);

  return List::create(
    Named("S_comp") = new_S_comp,
    Named("S_ens")  = new_S_ens,
    Named("alarm")  = alarm
  );
}