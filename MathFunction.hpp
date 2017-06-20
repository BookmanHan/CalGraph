#pragma once
#pragma once
#include "Import.hpp"
#include "Symbol.hpp"

namespace cal
{
	namespace math
	{
		class SymSigmoid
			:public Symbol
		{
		public:
			virtual void forward() override
			{
				value_forward = 1.f / (1.f + af::exp(-sym_in[0]->value_forward));
			}

			virtual void backward() override
			{
				sym_in[0]->value_backward = 
					value_backward 
					* value_forward 
					* (1.f - value_forward);
			}
		};

		Symbol& sigmoid(Symbol& a)
		{
			Symbol* node = new SymSigmoid;

			node->sym_in.push_back(&a);
			a.sym_out.push_back(node);

			return *node;
		}

		class SymTanh
			:public Symbol
		{
		public:
			virtual void forward() override
			{
				value_forward = af::tanh(sym_in[0]->value_forward);
			}

			virtual void backward() override
			{
				sym_in[0]->value_backward =
					value_backward
					* (1.f - value_forward * value_forward);
			}
		};

		Symbol& tanh(Symbol& a)
		{
			Symbol* node = new SymTanh;

			node->sym_in.push_back(&a);
			a.sym_out.push_back(node);

			return *node;
		}

		class SymAsin
			:public Symbol
		{
		public:
			virtual void forward() override
			{
				value_forward = af::asin(sym_in[0]->value_forward);
			}

			virtual void backward() override
			{
				sym_in[0]->value_backward =
					value_backward 
					/ af::sqrt(1.f - sym_in[0]->value_forward * sym_in[0]->value_forward);
			}
		};

		Symbol& arcsin(Symbol& a)
		{
			Symbol* node = new SymAsin;

			node->sym_in.push_back(&a);
			a.sym_out.push_back(node);

			return *node;
		}

		class SymAcos
			:public Symbol
		{
		public:
			virtual void forward() override
			{
				value_forward = af::acos(sym_in[0]->value_forward);
			}

			virtual void backward() override
			{
				sym_in[0]->value_backward =
					- value_backward
					/ af::sqrt(1.f - sym_in[0]->value_forward * sym_in[0]->value_forward);
			}
		};

		Symbol& arccos(Symbol& a)
		{
			Symbol* node = new SymAcos;

			node->sym_in.push_back(&a);
			a.sym_out.push_back(node);

			return *node;
		}

		class SymAtan
			:public Symbol
		{
		public:
			virtual void forward() override
			{
				value_forward = af::atan(sym_in[0]->value_forward);
			}

			virtual void backward() override
			{
				sym_in[0]->value_backward =
					value_backward / (1.f + sym_in[0]->value_forward * sym_in[0]->value_forward);
			}
		};

		Symbol& arctan(Symbol& a)
		{
			Symbol* node = new SymAtan;

			node->sym_in.push_back(&a);
			a.sym_out.push_back(node);

			return *node;
		}

		class SymSin
			:public Symbol
		{
		public:
			virtual void forward() override
			{
				value_forward = af::sin(sym_in[0]->value_forward);
			}

			virtual void backward() override
			{
				sym_in[0]->value_backward = value_backward * af::cos(sym_in[0]->value_forward);
			}
		};

		Symbol& sin(Symbol& a)
		{
			Symbol* node = new SymSin;

			node->sym_in.push_back(&a);
			a.sym_out.push_back(node);

			return *node;
		}

		class SymCos
			:public Symbol
		{
		public:
			virtual void forward() override
			{
				value_forward = af::cos(sym_in[0]->value_forward);
			}

			virtual void backward() override
			{
				sym_in[0]->value_backward = - value_backward * af::sin(sym_in[0]->value_forward);
			}
		};

		Symbol& cos(Symbol& a)
		{
			Symbol* node = new SymCos;

			node->sym_in.push_back(&a);
			a.sym_out.push_back(node);

			return *node;
		}

		class SymSinH
			:public Symbol
		{
		public:
			virtual void forward() override
			{
				value_forward = af::sinh(sym_in[0]->value_forward);
			}

			virtual void backward() override
			{
				sym_in[0]->value_backward =
					value_backward * af::cosh(sym_in[0]->value_forward);
			}
		};

		Symbol& sinh(Symbol& a)
		{
			Symbol* node = new SymSinH;

			node->sym_in.push_back(&a);
			a.sym_out.push_back(node);

			return *node;
		}

		class SymCosH
			:public Symbol
		{
		public:
			virtual void forward() override
			{
				value_forward = af::cosh(sym_in[0]->value_forward);
			}

			virtual void backward() override
			{
				sym_in[0]->value_backward =
						value_backward * af::sinh(sym_in[0]->value_forward);
			}
		};

		Symbol& cosh(Symbol& a)
		{
			Symbol* node = new SymCosH;

			node->sym_in.push_back(&a);
			a.sym_out.push_back(node);

			return *node;
		}

		class SymSqrt
			:public Symbol
		{
		public:
			virtual void forward() override
			{
				value_forward = af::sqrt(sym_in[0]->value_forward);
			}

			virtual void backward() override
			{
				sym_in[0]->value_backward =
					value_backward / value_forward / 2.f;
			}
		};

		Symbol& sqrt(Symbol& a)
		{
			Symbol* node = new SymSqrt;

			node->sym_in.push_back(&a);
			a.sym_out.push_back(node);

			return *node;
		}

		class SymExp
			:public Symbol
		{
		public:
			virtual void forward() override
			{
				value_forward = af::exp(sym_in[0]->value_forward);
			}

			virtual void backward() override
			{
				sym_in[0]->value_backward = value_backward * value_forward;
			}
		};

		Symbol& exp(Symbol& a)
		{
			Symbol* node = new SymExp;

			node->sym_in.push_back(&a);
			a.sym_out.push_back(node);

			return *node;
		}

		class SymLog
			:public Symbol
		{
		public:
			virtual void forward() override
			{
				value_forward = af::log(sym_in[0]->value_forward);
			}

			virtual void backward() override
			{
				sym_in[0]->value_backward =
						value_backward / value_forward;
			}
		};

		Symbol& log(Symbol& a)
		{
			Symbol* node = new SymLog;

			node->sym_in.push_back(&a);
			a.sym_out.push_back(node);

			return *node;
		}

		class SymNegate
			:public Symbol
		{
		public:
			virtual void forward() override
			{
				value_forward = - sym_in[0]->value_forward;
			}

			virtual void backward() override
			{
				sym_in[0]->value_backward = - value_backward;
			}
		};

		Symbol& neg(Symbol& a)
		{
			Symbol* node = new SymNegate;

			node->sym_in.push_back(&a);
			a.sym_out.push_back(node);

			return *node;
		}

		class SymABS
			:public Symbol
		{
		public:
			virtual void forward() override
			{
				value_forward = af::abs(sym_in[0]->value_forward);
			}

			virtual void backward() override
			{
				sym_in[0]->value_backward = af::sign(sym_in[0]->value_forward);
			}
		};

		Symbol& abs(Symbol& a)
		{
			Symbol* node = new SymABS;

			node->sym_in.push_back(&a);
			a.sym_out.push_back(node);

			return *node;
		}

		class SymSum
			:public Symbol
		{
		public:
			virtual void forward() override
			{
				value_forward = af::constant(af::sum<float>(sym_in[0]->value_forward), 
					sym_in[0]->value_forward.dims(0), sym_in[0]->value_forward.dims(1), 
					sym_in[0]->value_forward.dims(2), sym_in[0]->value_forward.dims(3));
			}

			virtual void backward() override
			{
				sym_in[0]->value_backward = value_backward;
			}
		};

		Symbol& sum(Symbol& a)
		{
			Symbol* node = new SymSum;

			node->sym_in.push_back(&a);
			a.sym_out.push_back(node);

			return *node;
		}

		class SymTransport
			:public Symbol
		{
		public:
			virtual void forward() override
			{
				value_forward = af::transpose(sym_in[0]->value_forward);
			}

			virtual void backward() override
			{
				sym_in[0]->value_backward = af::transpose(value_backward);
			}
		};

		Symbol& _t(Symbol& src)
		{
			Symbol* node = new SymTransport;

			node->sym_in.push_back(&src);
			src.sym_out.push_back(node);

			return *node;
		}
	}
}
