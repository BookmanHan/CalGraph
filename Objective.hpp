#pragma once
#include "Import.hpp"
#include "Symbol.hpp"
#include "Operation.hpp"
#include "MathFunction.hpp"

namespace cal
{
	using namespace math;
	namespace objective
	{
		Symbol& cross_entropi(Symbol& y, Symbol& target)
		{
			return y % log(target) + neg(y) % neg(target);
		}

		Symbol& softmax(Symbol& x)
		{
			return exp(x) / sum(exp(x));
		}
	}
}
