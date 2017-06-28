#pragma once
#include "Import.hpp"
#include "Symbol.hpp"
#include "Logging.hpp"

#define autoref auto&
namespace cal
{
	class CalGraph
	{
	public:
		vector<SymDatum*> DatumSources;
		vector<SymVariable*> Variables;
		vector<pair<Symbol*, string>> Losses;

	public:
		Symbol& datum(af::array& src)
		{
			auto node = new SymDatum;

			node->set(src);
			DatumSources.push_back(node);

			return *node;
		}

		Symbol& datum(int src)
		{
			auto node = new SymDatum;

			node->set(src);
			DatumSources.push_back(node);

			return *node;
		}

		Symbol& variable_embedding(af::array& src)
		{
			auto node = new SymEmbedding;

			node->set(src);
			Variables.push_back(node);

			return *node;
		}

		Symbol& variable(af::array& src)
		{
			auto node = new SymVariable;

			node->set(src);
			Variables.push_back(node);

			return *node;
		}

		Symbol& variable_normal(const int n_row, const int n_col)
		{
			auto node = new SymVariable;

			af::array mat(n_row, n_col);

			node->set(mat);
			Variables.push_back(node);

			return *node;
		}

		Symbol& variable_xavier(const int n_row, const int n_col)
		{
			auto node = new SymVariable;

			af::array mat = af::randn(n_row, n_col) / sqrt(n_row);

			node->set(mat);
			Variables.push_back(node);

			return *node;
		}

		Symbol& loss(Symbol& err, string name)
		{
			auto node = new SymLoss(err);

			Losses.push_back(make_pair(node, name));

			return *node;
		}

	public:
		void calculas()
		{
			for (auto i = DatumSources.begin(); i != DatumSources.end(); ++i)
			{
				(*i)->trigger_forward();
			}

			for (auto i = Variables.begin(); i != Variables.end(); ++i)
			{
				(*i)->trigger_forward();
			}
		}

		void update()
		{
			for (auto i = Losses.begin(); i != Losses.end(); ++i)
			{
				i->first->trigger_backward();
			}
		}

		void train(int epos, function<void(int epos)> batch = [](int) {})
		{
			while (epos-- > 0)
			{
				af::timer timer_f;
				timer_f.start();

				batch(epos);

				calculas();
				logout.record() << "\t[Forward] Time = " << timer_f.stop();

				logout.record() << "[CalGraph] Losses of " <<epos;
				for (auto i = Losses.begin(); i != Losses.end(); ++i)
				{
					float* value = i->first->value_forward.host<float>();
					logout.record() << "\tLoss of " << *value;
				}

				af::timer timer_b;
				timer_b.start();
				update();
				logout.record() << "\t[Backward] Time = " << timer_b.stop();
			}
		}
	};
}