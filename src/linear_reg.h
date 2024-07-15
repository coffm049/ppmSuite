#ifndef GIBBS_SAMPLER_H
#define GIBBS_SAMPLER_H

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>

void sample_beta_sigma2(const gsl_vector *y, const gsl_matrix *X, gsl_vector *beta, double *sigma2, const gsl_vector *beta0, const gsl_matrix *V0, double alpha0, double beta0_prior, gsl_rng *r, int iter);

#endif // GIBBS_SAMPLER_H

