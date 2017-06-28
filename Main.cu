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
	int n_sample = 3000;
	int n_feature = 100;
	int n_word = 100;
	int n_step = 20;
	int n_output = n_word;

	af::array input = af::randu(n_sample, n_step) * (n_word - 1);
	input = af::ceil(input);

	af::array output = af::randu(n_sample, n_step) * (n_word - 1);
	input = af::ceil(input);

	af::array hids = af::constant(0.f, n_sample, n_feature);
	af::array bitmask = af::constant(0, n_sample, n_feature, n_step);
	bitmask(af::span, af::span, af::seq(0, n_step/2)) = 1;

	CalGraph cg;

	autoref x = cg.datum(input);
	autoref y = cg.datum(output);
	autoref mask = cg.datum(bitmask);

	autoref Em = cg.variable_embedding(af::randn(n_word, n_feature));
	autoref W1 = cg.variable_xavier(n_feature, n_feature);
	autoref W2 = cg.variable_xavier(n_feature, n_feature);
	autoref W4 = cg.variable_xavier(n_feature, n_feature);
	autoref W5 = cg.variable_xavier(n_feature, n_feature);
	autoref W6 = cg.variable_xavier(n_feature, n_word);
	autoref W7 = cg.variable_xavier(n_feature, n_feature);

	auto hidden = &(cg.datum(hids));
	auto loss = &(cg.datum(af::constant(0.f, n_sample)));

	int n = 0;
	for(int i=0; i < n_step; ++i)
	{
		autoref step = cg.datum(n++);
		hidden = &(tanh(embed(Em, slice(2, x, step)) * W1 + (*hidden) * W2));
		hidden = &(*hidden % slice(3, mask, step));;
	}

	auto decoder_word = &(cg.datum(af::constant(0, n_sample, s32)));
	int m = 0;
	for (int i = 0; i < n_step; ++i)
	{
		autoref step = cg.datum(m++);
		hidden = &(tanh(*hidden * W7 + embed(Em, *decoder_word) * W4));
		autoref prob = softmax(*hidden * W6);
		decoder_word = &(max_index(*hidden));
		loss = &(*loss + 
			scalar_sum(cross_entropi(hoc(n_sample, n_word, slice(2, x, step)), prob)));
	}

	cg.loss(*loss, "RNN");

	cg.train(10000,
		[&](int epos)
	{
		x.set(input);
		y.set(output);
	});


	return 0;
}