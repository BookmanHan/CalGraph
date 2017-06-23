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
	int n_sample = 10000;
	int n_feature = 100;
	int n_step = 20;

	af::array input = af::randu(n_sample, n_step) * 10.0f;
	input = af::ceil(input);

	af::array output = af::randn(n_sample, 2);
	output(af::seq(0, af::end, 2), af::span) = 0.9;
	output(af::seq(1, af::end, 2), af::span) = 0.1;

	af::array hids = af::constant(0.f, n_sample, n_feature);
	af::array bitmask = af::constant(0, n_sample, n_feature, 20, u8);
	bitmask(af::span, af::span, af::seq(0, 15)) = 1;

	CalGraph cg;

	autoref x = cg.datum(input);
	autoref y = cg.datum(output);
	autoref mask = cg.datum(bitmask);

	autoref Em = cg.variable_embedding(af::randn(20, 100));
	autoref W1 = cg.variable_xavier(n_feature, n_feature);
	autoref W2 = cg.variable_xavier(n_feature, n_feature);
	autoref W3 = cg.variable_xavier(n_feature, 2);
	autoref W4 = cg.variable_xavier(n_feature, n_feature);
	autoref W5 = cg.variable_xavier(n_feature, n_feature);

	auto hidden = &(cg.datum(hids));
	auto loss = &(cg.datum(af::constant(0.f, n_sample, 2)));

	int n = 0;
	for(int i=0; i<20; ++i)
	{
		autoref step = cg.datum(n++);
		hidden = &(tanh(embed(Em, slice(2, x, step)) * W1 + (*hidden) * W2));
		hidden = &(*hidden % slice(3, mask, step));;
	}

	int m = 0;
	for (int i = 0; i < 20; ++i)
	{
		autoref step = cg.datum(m++);
		autoref rep = softmax(((*hidden) * W5 + embed(Em, slice(2, x, step)) * W4) * W3);
		loss = &(*loss + cross_entropi(rep, y));
	}

	cg.loss(*loss, "RNN");

	try
	{
		cg.train(10000,
			[&](int epos)
		{
			x.set(input);
			y.set(output);
		});
	}
	catch (af::exception e)
	{
		cout << e.what() << endl;
	}

	return 0;
}