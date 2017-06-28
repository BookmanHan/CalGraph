#pragma once
#pragma once
#include "Import.hpp"
#include "Symbol.hpp"

namespace cal
{
	namespace logic
	{
		class SymMaxIndex
			:public Symbol
		{
		public:
			virtual void forward() override
			{
				af::array cont;
				af::max(cont, value_forward, sym_in[0]->value_forward, 1);
			}

			virtual void backward() override
			{
				;
			}

		public:
			virtual void trigger_backward()
			{
				;
			}
		};

		Symbol& max_index(Symbol& src)
		{
			Symbol* node = nullptr;
			node = new SymMaxIndex;

			node->sym_in.push_back(&src);
			src.sym_out.push_back(node);

			return *node;
		}

		class SymHoc
			:public Symbol
		{
		public:
			int m;
			int n;
		public:
			virtual void forward() override
			{
				value_forward = af::constant(0.f, n, m);
				af::array bi = af::range(m);
				value_forward(sym_in[0]->value_forward + n * bi) = 1.f;
				value_forward = transpose(value_forward);
			}

			virtual void backward() override
			{
				;
			}

		public:
			virtual void trigger_backward()
			{
				;
			}
		};

		Symbol& hoc(int m, int n, Symbol& a)
		{
			Symbol* node = nullptr;
			node = new SymHoc;
			((SymHoc*)node)->m = m;
			((SymHoc*)node)->n = n;

			node->sym_in.push_back(&a);
			a.sym_out.push_back(node);

			return *node;
		}
	}
}