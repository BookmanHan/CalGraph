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
			return
				y % log(target)
				+ neg(y) % log(neg(target));
		}

		Symbol& softmax(Symbol& x)
		{
			return exp(x) / sum(exp(x));
		}

		class SymJoinStep
			:public Symbol
		{
		public:
			virtual void forward() override
			{
				value_forward = af::constant(0.f, 0);
				for (auto i = sym_in.begin(); i != sym_in.end(); ++i)
				{
					value_forward = af::join(2, value_forward, (*i)->value_forward);
				}
			}

			//virtual void backward() override
			//{
			//	for (auto i = sym_in.begin(); i != sym_in.end(); ++i)
			//	{
			//		(*i)->value_backward = 
			//			value_backward(af::span, af::span, i - sym_in.begin());
			//	}
			//}
		public:
			virtual void trigger_backward()
			{
				;
			}
		};

		Symbol& join_step(vector<Symbol*>& src)
		{
			Symbol* node = new SymJoinStep;

			for (auto i = src.begin(); i != src.end(); ++i)
			{
				node->sym_in.push_back(*i);
				(*i)->sym_out.push_back(node);
			}

			return *node;
		}


		class SymWeighting
			:public Symbol
		{
		public:
			virtual void forward() override
			{
				value_forward = af::constant(
					0.f, 
					sym_in[0]->value_forward.dims(0), 
					sym_in[0]->value_forward.dims(1));
				af::array weight = af::moddims(
					sym_in[1]->value_forward, 
					sym_in[1]->value_forward.dims(0), 
					1, 
					sym_in[1]->value_forward.dims(1));

				logout.record() << weight.dims();
				logout.record() << sym_in[0]->value_forward.dims();
				for(auto i = 0; i<sym_in[0]->value_forward.dims(1); ++i)
				{
					value_forward(af::span, i)= 
						af::sum(weight * sym_in[0]->value_forward(af::span, i, af::span), 2);
				}
			}

			virtual void backward() override
			{
				sym_in[1]->value_backward = af::constant(0.f, sym_in[1]->value_forward.dims());
				for(int i=0; i<sym_in[1]->value_forward.dims(0); ++i)
				{
					sym_in[1]->value_backward(i, af::span) =
						af::matmul(
							value_backward(i, af::span),
								af::moddims(
									sym_in[0]->value_forward(i, af::span, af::span),
									sym_in[0]->value_forward.dims(1),
									sym_in[0]->value_forward.dims(2)));
				}

				//sym_in[0]->value_backward = af::constant(0.f, sym_in[0]->value_forward.dims());
				//for(int i=0; i<sym_in[0]->value_forward.dims(0); ++i)
				//{
				//	sym_in[0]->value_backward(i, af::span, af::span) =
				//		af::moddims(
				//			af::matmul(
				//				transpose(value_backward(i, af::span)),
				//				sym_in[1]->value_forward(i, af::span)),
				//			1, 
				//			sym_in[0]->value_backward.dims(1),
				//			sym_in[0]->value_backward.dims(2));
				//}
			}
		};

		Symbol& weight_step(Symbol& weight, Symbol& join)
		{
			Symbol* node = new SymWeighting;

			node->sym_in.push_back(&join);
			node->sym_in.push_back(&weight);
			weight.sym_out.push_back(node);
			join.sym_out.push_back(node);

			return *node;
		}
	}
}
