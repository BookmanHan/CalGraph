#pragma once
#include "Import.hpp"

namespace cal
{
	namespace Solver
	{
		class DecentAdaDelta
		{
		protected:
			const float moment;
			const float regularization;

		public:
			DecentAdaDelta(float moment, float regularization)
				:moment(moment), regularization(regularization)
			{
				;
			}

		public:
			void gradient(float& derv_grad, float& derv_x, float& elem, float& grad) const
			{
				derv_grad = moment * derv_grad + (1 - moment) * grad * grad;
				float derv_elem = sqrt(derv_x + regularization) / sqrt(derv_grad + regularization) * grad;
				derv_x = moment * derv_x + (1 - moment) * derv_elem * derv_elem;

				elem -= derv_elem;
			}

			void gradient(
				af::array& derv_grad,
				af::array& derv_x,
				af::array& elem,
				af::array& grad) const
			{
				derv_grad = moment * derv_grad + (1 - moment) * grad * grad;

				derv_grad = moment * derv_grad + (1 - moment) * grad * grad;
				af::array derv_elem = af::sqrt(derv_x + regularization) 
					/ af::sqrt(derv_grad + regularization) * grad;
				derv_x = moment * derv_x + (1 - moment) * derv_elem * derv_elem;

				elem -= derv_elem;
			}
		};

		DecentAdaDelta* global_calc_graph_solver = new DecentAdaDelta(0.6, 1e-6);

		inline
		void SetupAdaDeltaSolver(float moment, float regularization)
		{
			delete global_calc_graph_solver;
			global_calc_graph_solver = new DecentAdaDelta(moment, regularization);
		}
	}
}