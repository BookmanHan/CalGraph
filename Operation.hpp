#pragma once
#include "Import.hpp"
#include "Symbol.hpp"

namespace cal
{
	class SymPlus
		:public Symbol
	{
	public:
		virtual void forward() override
		{
			value_forward = sym_in[0]->value_forward + sym_in[1]->value_forward;
		}

		virtual void backward() override
		{
			sym_in[0]->value_backward = value_backward;
			sym_in[1]->value_backward = value_backward;
		}
	};

	Symbol& operator+(Symbol& a, Symbol&b)
	{
		Symbol* node = new SymPlus;

		node->sym_in.push_back(&a);
		node->sym_in.push_back(&b);
		a.sym_out.push_back(node);
		b.sym_out.push_back(node);

		return *node;
	}

	class SymMinus
		:public Symbol
	{
	public:
		virtual void forward() override
		{
			value_forward = sym_in[0]->value_forward - sym_in[1]->value_forward;
		}

		virtual void backward() override
		{
			sym_in[0]->value_backward = value_backward;
			sym_in[1]->value_backward = -value_backward;
		}
	};

	Symbol& operator-(Symbol& a, Symbol&b)
	{
		Symbol* node = new SymMinus;

		node->sym_in.push_back(&a);
		node->sym_in.push_back(&b);
		a.sym_out.push_back(node);
		b.sym_out.push_back(node);

		return *node;
	}

	class SymMultiply
		:public Symbol
	{
	public:
		virtual void forward() override
		{
			value_forward = af::matmul(sym_in[0]->value_forward, sym_in[1]->value_forward);
		}

		virtual void backward() override
		{
			sym_in[0]->value_backward = af::matmul(value_backward, af::transpose(sym_in[1]->value_forward));
			sym_in[1]->value_backward = af::matmul(af::transpose(sym_in[0]->value_forward), value_backward);
		}
	};

	Symbol& operator*(Symbol& a, Symbol& b)
	{
		Symbol* node = new SymMultiply;

		node->sym_in.push_back(&a);
		node->sym_in.push_back(&b);
		a.sym_out.push_back(node);
		b.sym_out.push_back(node);

		return *node;
	}

	class SymDivide
		:public Symbol
	{
	public:
		virtual void forward() override
		{
			value_forward = sym_in[0]->value_forward / sym_in[1]->value_forward;
		}

		virtual void backward() override
		{
			sym_in[0]->value_backward = value_backward / sym_in[1]->value_forward;
			sym_in[1]->value_backward =
				value_backward * sym_in[0]->value_forward
				/ sym_in[1]->value_forward / sym_in[1]->value_forward;
		}
	};

	Symbol& operator/(Symbol& a, Symbol& b)
	{
		Symbol* node = new SymDivide;

		node->sym_in.push_back(&a);
		node->sym_in.push_back(&b);
		a.sym_out.push_back(node);
		b.sym_out.push_back(node);

		return *node;
	}

	class SymPairMultiply
		:public Symbol
	{
	public:
		virtual void forward() override
		{
			value_forward = sym_in[0]->value_forward * sym_in[1]->value_forward;
		}

		virtual void backward() override
		{
			sym_in[0]->value_backward = value_backward * sym_in[1]->value_forward;
			sym_in[1]->value_backward = sym_in[0]->value_forward * value_backward;
		}
	};

	Symbol& operator%(Symbol& a, Symbol& b)
	{
		Symbol* node = new SymPairMultiply;

		node->sym_in.push_back(&a);
		node->sym_in.push_back(&b);
		a.sym_out.push_back(node);
		b.sym_out.push_back(node);

		return *node;
	}

	class SymPower
		:public Symbol
	{
	public:
		virtual void forward() override
		{
			value_forward = af::pow(
				sym_in[0]->value_forward,
				sym_in[1]->value_forward);
		}

		virtual void backward() override
		{
			sym_in[0]->value_backward =
				value_backward
				* sym_in[1]->value_forward
				* af::pow(
					sym_in[0]->value_forward,
					sym_in[1]->value_forward - 1.f);

			sym_in[1]->value_backward =
				value_backward
				* value_forward
				* af::log(sym_in[0]->value_forward);
		}
	};

	Symbol& operator^(Symbol& a, Symbol& b)
	{
		Symbol* node = new SymPower;

		node->sym_in.push_back(&a);
		node->sym_in.push_back(&b);
		a.sym_out.push_back(node);
		b.sym_out.push_back(node);

		return *node;
	}

	class SymView1D
		:public Symbol
	{
	public:
		virtual void forward() override
		{
			int* value_index = sym_in[1]->value_forward(0).host<int>();
			value_forward = sym_in[0]->value_forward(*value_index, af::span, af::span, af::span);
		}

		virtual void backward() override
		{
			int* value_index = sym_in[1]->value_forward(0).host<int>();
			if (sym_in[0]->is_datum() == false)
				sym_in[0]->value_backward(*value_index, af::span, af::span, af::span) = value_backward;
		}
	};

	class SymView2D
		:public Symbol
	{
	public:
		virtual void forward() override
		{
			int* value_index = sym_in[1]->value_forward(0).host<int>();
			value_forward = sym_in[0]->value_forward(af::span, *value_index, af::span, af::span);
		}

		virtual void backward() override
		{
			int* value_index = sym_in[1]->value_forward(0).host<int>();
			if (sym_in[0]->is_datum() == false)
				sym_in[0]->value_backward(af::span, *value_index, af::span, af::span) = value_backward;
		}
	};

	class SymView3D
		:public Symbol
	{
	public:
		virtual void forward() override
		{
			int* value_index = sym_in[1]->value_forward(0).host<int>();
			value_forward = sym_in[0]->value_forward(af::span, af::span, *value_index, af::span);
		}

		virtual void backward() override
		{
			int* value_index = sym_in[1]->value_forward(0).host<int>();
			if (sym_in[0]->is_datum() == false)
				sym_in[0]->value_backward(af::span, af::span, *value_index, af::span) = value_backward;
		}
	};

	class SymView4D
		:public Symbol
	{
	public:
		virtual void forward() override
		{
			int* value_index = sym_in[1]->value_forward(0).host<int>();
			value_forward = sym_in[0]->value_forward(af::span, af::span, af::span, *value_index);
		}

		virtual void backward() override
		{
			int* value_index = sym_in[1]->value_forward(0).host<int>();
			if (sym_in[0]->is_datum() == false)
				sym_in[0]->value_backward(af::span, af::span, af::span, *value_index) = value_backward;
		}
	};

	Symbol& slice(int dim, Symbol& src, Symbol& index)
	{
		Symbol* node = nullptr;

		switch (dim)
		{
		case 1:
			node = new SymView1D;
			break;
		case 2:
			node = new SymView2D;
			break;
		case 3:
			node = new SymView3D;
			break;
		case 4:
			node = new SymView4D;
			break;
		default:
			throw af::exception("Slice(): Dimension is incorrect.");
			break;
		}

		node->sym_in.push_back(&src);
		node->sym_in.push_back(&index);
		src.sym_out.push_back(node);
		index.sym_out.push_back(node);

		return *node;
	}

	Symbol& embed(Symbol& src, Symbol& ind)
	{
		Symbol* node = new SymEmbeddingNode;

		node->sym_in.push_back(&src);
		node->sym_in.push_back(&ind);
		src.sym_out.push_back(node);
		ind.sym_out.push_back(node);

		return *node;
	}
}