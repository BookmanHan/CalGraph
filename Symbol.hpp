#pragma once
#include "Import.hpp"
#include "Solver.hpp"

namespace cal
{
	class Symbol
	{
	public:
		int cur_sym_in;
		vector<Symbol*>	sym_in;
		vector<Symbol*>	sym_out;

	public:
		Symbol()
			:cur_sym_in(0)
		{
			;
		}

	public:
		af::array value_forward;
		af::array value_backward;
		af::array value_backward_x;
		af::array value_backward_grad;

	public:
		virtual void forward()
		{
			;
		}

		virtual void backward()
		{
			;
		}

	public:
		af::array& data()
		{
			return value_forward;
		}

	public:
		virtual void set(int index)
		{
			value_forward = af::constant(index, 1, s32);
		}

		virtual void set(af::array& content)
		{
			value_forward = content;
		}

		void reset_gradient()
		{
			value_backward_grad = af::constant(0.f,
				value_forward.dims(0),
				value_forward.dims(1),
				value_forward.dims(2),
				value_forward.dims(3));
			value_backward_x = af::constant(0.f,
				value_forward.dims(0),
				value_forward.dims(1),
				value_forward.dims(2),
				value_forward.dims(3));
			value_backward = af::constant(0.f,
				value_forward.dims(0),
				value_forward.dims(1),
				value_forward.dims(2),
				value_forward.dims(3));
		}

	public:
		void trigger_forward()
		{
			++cur_sym_in;
			if (cur_sym_in >= sym_in.size())
			{
				forward();
				for (auto i = sym_out.begin(); i != sym_out.end(); ++i)
				{
					(*i)->trigger_forward();
				}
				cur_sym_in = 0;
			}
		}

		void trigger_backward()
		{
			backward();
			for (auto i = sym_in.begin(); i != sym_in.end(); ++i)
			{
				(*i)->trigger_backward();
			}
		}

	public:
		virtual bool is_datum()
		{
			return false;
		}

		virtual bool is_variable()
		{
			return false;
		}

		virtual bool is_loss()
		{
			return false;
		}
	};

	class SymVariable
		:public Symbol
	{
	public:
		SymVariable()
		{

		}

		SymVariable(vector<Symbol*>& inputs)
		{
			inputs.push_back(this);
		}

	public:
		virtual void set(int index)
		{
			value_forward = af::constant(index, 1, s32);
			reset_gradient();
		}

		virtual void set(af::array& content)
		{
			value_forward = content;
			reset_gradient();
		}

	public:
		virtual void backward() override
		{
			cal::Solver::global_calc_graph_solver->gradient(value_backward_grad, value_backward_x, value_forward, value_backward);
		}

	public:
		virtual bool is_variable()
		{
			return true;
		}
	};

	class SymEmbedding
		:public SymVariable
	{
	public:
		SymEmbedding()
		{

		}

		SymEmbedding(vector<Symbol*>& inputs)
		{
			inputs.push_back(this);
		}

	public:
		virtual void set(int index)
		{
			value_forward = af::constant(index, 1, s32);
			reset_gradient();
		}

		virtual void set(af::array& content)
		{
			value_forward = content;
			reset_gradient();
		}

	public:
		virtual bool is_variable()
		{
			return true;
		}
	};

	class SymEmbeddingNodeUpdatable
		:public Symbol
	{
	public:
		virtual void forward() override
		{
			value_forward = sym_in[0]->value_forward(af::flat(sym_in[1]->value_forward), af::span, af::span, af::span);
		}

		virtual void backward() override
		{
			cal::Solver::global_calc_graph_solver->gradient
			(sym_in[0]->value_backward_grad,
				sym_in[0]->value_backward_x,
				sym_in[0]->value_forward,
				value_backward,
				af::flat(sym_in[1]->value_forward));
		}
	};

	class SymEmbeddingNodeFixed
		:public Symbol
	{
	public:
		virtual void forward() override
		{
			value_forward = sym_in[0]->value_forward(af::flat(sym_in[1]->value_forward), af::span, af::span, af::span);
		}

		virtual void backward() override
		{
			;
		}
	};

	class SymLoss
		:public Symbol
	{
	public:
		SymLoss(Symbol& node)
		{
			sym_in.push_back(&node);
			node.sym_out.push_back(this);
		}

	public:
		virtual void forward() override
		{
			value_forward = af::constant(af::sum<float>(sym_in[0]->value_forward), 1, 1);
		}

		virtual void backward() override
		{
			value_backward = af::constant(1, 1, 1.f);
			sym_in[0]->value_backward = af::constant(1.f,
				sym_in[0]->value_forward.dims(0),
				sym_in[0]->value_forward.dims(1),
				sym_in[0]->value_forward.dims(2),
				sym_in[0]->value_forward.dims(3));
		}

	public:
		virtual bool is_loss()
		{
			return true;
		}
	};

	class SymDatum
		:public Symbol
	{
	public:
		SymDatum()
		{
			reset_gradient();
		}

		SymDatum(vector<Symbol*>& inputs)
		{
			reset_gradient();
			inputs.push_back(this);
		}

	public:
		virtual bool is_datum()
		{
			return true;
		}
	};

	class SymConst
		:public Symbol
	{
	public:
		SymConst()
		{
			;
		}

		SymConst(vector<Symbol*>& inputs)
		{
			inputs.push_back(this);
		}
	};

	Symbol& _c(const af::array& const_mat)
	{
		Symbol* node = new SymConst;
		node->value_forward = const_mat;

		return *node;
	}

	class SymPrint
		:public Symbol
	{
	public:
		virtual void forward() override
		{
			value_forward = sym_in[0]->value_forward;
			af_print(value_forward);
		}

		virtual void backward() override
		{
			sym_in[0]->value_backward = value_backward;
			af_print(value_backward);
		}
	};

	Symbol& print(Symbol& src)
	{
		Symbol* node = new SymPrint;

		node->sym_in.push_back(&src);
		src.sym_out.push_back(node);

		return *node;
	}
}