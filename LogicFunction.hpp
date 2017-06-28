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
		};

		Symbol& max_index(Symbol& src)
		{
			Symbol* node = nullptr;
			node = new SymMaxIndex;

			node->sym_in.push_back(&src);
			src.sym_out.push_back(node);

			return *node;
		}
	}
}