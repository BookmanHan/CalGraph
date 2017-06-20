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
	af::array input = af::randn(10, 10);
	input(af::seq(0, af::end, 2), af::span) += 10.0f;

	af::array output = af::randn(10, 2);
	output(af::seq(0, af::end, 2), af::span) = 0.9;
	output(af::seq(1, af::end, 2), af::span) = 0.1;

	af::array hids = af::constant(0.f, 10, 10);
	af::array bitmask = af::constant(0, 10, 10, 30, u8);
	bitmask(af::span, af::span, af::seq(0, 20)) = 1;
	
	CalGraph cg;

	autoref x = cg.datum(input);
	autoref y = cg.datum(output);
	autoref mask = cg.datum(bitmask);

	autoref W1 = cg.variable_xavier(10, 10);
	autoref W2 = cg.variable_xavier(10, 10);
	autoref W3 = cg.variable_xavier(10, 2);

	auto hidden = &(cg.datum(hids));
	auto loss = &(cg.datum(af::constant(0.f, 10, 2)));

	int n = 0;
	for(int i=0; i<20; ++i)
	{
		//autoref step = cg.datum(n++);
		hidden = &(tanh(x * W1 + (*hidden) * W2));
		//hidden = &(*hidden % slice(3, mask, step));

		autoref rep = (*hidden) * W3;

		loss = &(*loss + (rep - y) % (rep - y));
	}

	cg.loss(*loss, "RNN");
	cg.train(10000);

	return 0;
}