#include "Import.hpp"
#include "CalGraph.hpp"
#include "MathFunction.hpp"
#include "Operation.hpp"
#include "Objective.hpp"

using namespace cal;
using namespace cal::math;
using namespace cal::objective;

int main(int argc, char **argv)
{
	int n_sample = 1000;
	int n_feature = 100;

	af::array input = af::randn(n_sample, n_feature);
	input(af::seq(0, af::end, 2), af::span) += 10.0f;

	af::array output = af::randn(n_sample, 2);
	output(af::seq(0, af::end, 2), af::span) = 0.9;
	output(af::seq(1, af::end, 2), af::span) = 0.1;

	af::array hids = af::constant(0.f, n_sample, n_feature);
	
	CalGraph cg;

	autoref x = cg.datum(input);
	autoref y = cg.datum(output);

	autoref W1 = cg.variable_xavier(n_feature, n_feature);
	autoref W2 = cg.variable_xavier(n_feature, n_feature);
	autoref W3 = cg.variable_xavier(n_feature, 2);

	auto hidden = &(cg.datum(hids));
	auto loss = &(cg.datum(af::constant(0.f, n_sample, 2)));

	int n = 0;
	for(int i=0; i<15; ++i)
	{
		hidden = &(tanh(x * W1 + (*hidden) * W2));
		autoref rep = softmax((*hidden) * W3);
		loss = &(*loss + cross_entropi(rep, y));
	}

	cg.loss(*loss, "RNN");
	cg.train(10000);

	return 0;
}