// =============================================================================
//  CADET
//  
//  Copyright © 2008-2024: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

#include "model/binding/BindingModelBase.hpp"
#include "model/ExternalFunctionSupport.hpp"
#include "model/ModelUtils.hpp"
#include "cadet/Exceptions.hpp"
#include "model/Parameters.hpp"
#include "LocalVector.hpp"
#include "SimulationTypes.hpp"

#include <cmath>
#include <functional>
#include <unordered_map>
#include <string>
#include <vector>

/*<codegen>
{
	"name": "LinearLSSParamHandler",
	"externalName": "ExtLinearLSSParamHandler",
	"parameters":
		[
			{ "type": "ScalarComponentDependentParameter", "varName": "kkin", "confName": "LINLSS_KKIN"},
			{ "type": "ScalarComponentDependentParameter", "varName": "a0", "confName": "LINLSS_A0"},
			{ "type": "ScalarComponentDependentParameter", "varName": "ss", "confName": "LINLSS_SS"},
			{ "type": "ScalarComponentDependentParameter", "varName": "ct", "confName": "LINLSS_CT"},
			{ "type": "ScalarComponentDependentParameter", "varName": "T", "confName": "LINLSS_T"}
		]
}
</codegen>*/


/* Parameter description
 ------------------------
 kkin = Linear driving force
 a0 = Henry Coefficient of component in pure water
 ss = Solvent strength parameter
 ct = Undefined
 T = Temperature

 q = a0 * exp(-ss * phi) * exp(ct * 1000 * (1/T - 1/Tref))

 phi is controlled by yCp[0]
 Tref is defined as 298.15 K
*/

namespace cadet
{

	namespace model
	{
		// Configure the name of the binding model
		inline const char* LinearLSSParamHandler::identifier() CADET_NOEXCEPT { return "LINEAR_LSS"; }

		// Check if user input correct number of entries in the parameter list
		inline bool LinearLSSParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
		{
			// Note the "_" is automatically added by code to the defined variables

			if ((_a0.size() != _ss.size()) || (_a0.size() != _ct.size()) || (_a0.size() != _T.size()) || (_a0.size() != _kkin.size()) || (_a0.size() < nComp))
				throw InvalidParameterException("LINLSS_KKIN, LINLSS_A0, LINLSS_SS, LINLSS_CT, and LINLSS_T have to have the same size");

			return true;
		}

		// Configure the name of the binding model with the external function
		inline const char* ExtLinearLSSParamHandler::identifier() CADET_NOEXCEPT { return "EXT_LINEAR_LSS"; }

		// Check if user input correct number of entries in the parameter list
		inline bool ExtLinearLSSParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
		{
			if ((_a0.size() != _ss.size()) || (_a0.size() != _ct.size()) || (_a0.size() != _T.size()) || (_a0.size() != _kkin.size()) || (_a0.size() < nComp))
				throw InvalidParameterException("LINLSS_KKIN, LINLSS_A0, LINLSS_SS, LINLSS_CT, and LINLSS_T have to have the same size");

			return true;
		}


		template <class ParamHandler_t>
		class LinearLSSBindingBase : public ParamHandlerBindingModelBase<ParamHandler_t>
		{
		public:

			LinearLSSBindingBase() { }
			virtual ~LinearLSSBindingBase() CADET_NOEXCEPT { }

			static const char* identifier() { return ParamHandler_t::identifier(); }

			// Change to false to use Algorithmic Differentiation for Jacobian for testing
			virtual bool implementsAnalyticJacobian() const CADET_NOEXCEPT { return true; }

			CADET_BINDINGMODELBASE_BOILERPLATE

		protected:
			using ParamHandlerBindingModelBase<ParamHandler_t>::_paramHandler;
			using ParamHandlerBindingModelBase<ParamHandler_t>::_reactionQuasistationarity;
			using ParamHandlerBindingModelBase<ParamHandler_t>::_nComp;
			using ParamHandlerBindingModelBase<ParamHandler_t>::_nBoundStates;

			template <typename StateType, typename CpStateType, typename ResidualType, typename ParamType>
			int fluxImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, StateType const* y,
				CpStateType const* yCp, ResidualType* res, LinearBufferAllocator workSpace) const
			{
				typename ParamHandler_t::ParamsHandle const p = _paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);


				unsigned int bndIdx = 0;
				for (int i = 0; i < _nComp; ++i)
				{
					// Skip components without bound states (bound state index bndIdx is not advanced)
					if (_nBoundStates[i] == 0)
						continue;

					// Residual
					res[bndIdx] = static_cast<ParamType>(p->kkin[i]) * (y[bndIdx] - static_cast<ParamType>(p->a0[i]) * exp(-static_cast<ParamType>(p->ss[i]) * yCp[0]) * exp(static_cast<ParamType>(p->ct[i]) * 1000 * (1 / static_cast<ParamType>(p->T[i]) - 1 / 298.15)) * yCp[i]);
					// res[bndIdx] = -static_cast<ParamType>(p->a0[i]) * yCp[i] + static_cast<ParamType>(p->ss[i]) * y[bndIdx];

					// Next bound component
					++bndIdx;
				}

				return 0;
			}

			template <typename RowIterator>
			void jacobianImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double const* yCp, int offsetCp, RowIterator jac, LinearBufferAllocator workSpace) const
			{
				typename ParamHandler_t::ParamsHandle const p = _paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

				int bndIdx = 0;
				for (int i = 0; i < _nComp; ++i)
				{
					// Skip components without bound states (bound state index bndIdx is not advanced)
					if (_nBoundStates[i] == 0)
						continue;

					jac[0] += static_cast<double>(p->kkin[i]); // dres / dq_i
					jac[i - bndIdx - offsetCp] = -static_cast<double>(p->kkin[i]) * static_cast<double>(p->a0[i]) * exp(-static_cast<double>(p->ss[i]) * yCp[0]) * exp(static_cast<double>(p->ct[i]) * 1000 * (1 / static_cast<double>(p->T[i]) - 1 / 298.15)); // dres / dc_{p,i}

					++bndIdx;
					++jac;
				}
			}
		};

		typedef LinearLSSBindingBase<LinearLSSParamHandler> LinearLSSBinding;
		typedef LinearLSSBindingBase<ExtLinearLSSParamHandler> ExternalLinearLSSBinding;

		namespace binding
		{
			void registerLinearLSSModel(std::unordered_map<std::string, std::function<model::IBindingModel* ()>>& bindings)
			{
				bindings[LinearLSSBinding::identifier()] = []() { return new LinearLSSBinding(); };
				bindings[ExternalLinearLSSBinding::identifier()] = []() { return new ExternalLinearLSSBinding(); };
			}
		}  // namespace binding

	}  // namespace model

}  // namespace cadet