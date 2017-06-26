#include "Import.hpp"
#include "CalGraph.hpp"
#include "MathFunction.hpp"
#include "LogicFunction.hpp"
#include "Operation.hpp"
#include "Objective.hpp"

using namespace cal;
using namespace cal::math;
using namespace cal::objective;
using namespace cal::logic;

int main(int argc, char **argv)
{
	int n_sample = 1000;
	int n_feature = 100;
	int n_word = 100;
	int n_step = 20;
	int n_output = 2;

	af::array input = af::randu(n_sample, n_step) * (n_word - 1);
	input = af::ceil(input);

	af::array output = af::randu(n_sample, n_output);
	output(output > 0.5) = 0.95;
	output(output < 0.5) = 0.05;

	af::array hids = af::constant(0.f, n_sample, n_feature);
	af::array bitmask = af::constant(0, n_sample, n_feature, n_step, u8);
	bitmask(af::span, af::span, af::seq(0, 15)) = 1;

	CalGraph cg;

	autoref x = cg.datum(input);
	autoref y = cg.datum(output);
	autoref mask = cg.datum(bitmask);

	autoref Em = cg.variable_embedding(af::randn(n_word, n_feature));
	autoref W1 = cg.variable_xavier(n_feature, n_feature);
	autoref W2 = cg.variable_xavier(n_feature, n_feature);
	autoref W3 = cg.variable_xavier(n_word, 2);
	autoref W4 = cg.variable_xavier(n_feature, n_feature);
	autoref W5 = cg.variable_xavier(n_feature, n_feature);
	autoref W6 = cg.variable_xavier(n_feature, n_word);
	autoref W7 = cg.variable_xavier(n_feature, n_feature);

	auto hidden = &(cg.datum(hids));
	auto loss = &(cg.datum(af::constant(0.f, n_sample, 2)));

	int n = 0;
	for(int i=0; i < 5; ++i)
	{
		autoref step = cg.datum(n++);
		hidden = &(tanh(embed(Em, slice(2, x, step)) * W1 + (*hidden) * W2));
		hidden = &(*hidden % slice(3, mask, step));;
	}

	auto decoder_word = &(cg.datum(af::constant(0, n_sample, s32)));
	int m = 0;
	for (int i = 0; i < 5; ++i)
	{
		autoref step = cg.datum(m++);
		hidden = &(*hidden * W7 + embed(Em, *decoder_word) * W4);
		autoref prob = *hidden * W6;
		decoder_word = &(max_index(prob));
		print(*decoder_word);

		loss = &(*loss + cross_entropi(prob*W3, y));
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