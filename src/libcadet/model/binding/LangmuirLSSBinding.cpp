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
	"name": "LangmuirLSSParamHandler",
	"externalName": "ExtLangmuirLSSParamHandler",
	"parameters":
		[
			{ "type": "ScalarComponentDependentParameter", "varName": "kkin", "confName": "MCLLSS_KKIN"},
			{ "type": "ScalarComponentDependentParameter", "varName": "qMax", "confName": "MCLLSS_QMAX"},
			{ "type": "ScalarComponentDependentParameter", "varName": "b0", "confName": "MCLLSS_B0"},
			{ "type": "ScalarComponentDependentParameter", "varName": "ss", "confName": "MCLLSS_SS"},
			{ "type": "ScalarComponentDependentParameter", "varName": "ct", "confName": "MCLLSS_CT"},
			{ "type": "ScalarComponentDependentParameter", "varName": "T", "confName": "MCLLSS_T"}
		]
}
</codegen>*/

/* Parameter description
 ------------------------
 kkin = Linear driving force
 qMax = Capacity
 b0 = Equilibrium constant of component in pure water
 ss = Solvent strength parameter
 ct = Temperature coefficient
 T = Temperature

 q = (qmax * b * c) / (1 + b * c)
 b = b0 * exp(-ss * phi) * exp(ct * 1000 * (1/T - 1/Tref))

 phi is controlled by yCp[0]
 Tref is defined as 298.15 K
*/

namespace cadet
{

	namespace model
	{

		inline const char* LangmuirLSSParamHandler::identifier() CADET_NOEXCEPT { return "MULTI_COMPONENT_LANGMUIR_LSS"; }

		inline bool LangmuirLSSParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
		{
			if ((_kkin.size() != _qMax.size()) || (_kkin.size() != _b0.size()) || (_kkin.size() != _ss.size()) || (_kkin.size() != _ct.size()) || (_kkin.size() != _T.size()) || (_kkin.size() < nComp))
				throw InvalidParameterException("MCLLSS_KKIN, MCLLSS_QMAX, MCLLSS_B0, MCLLSS_SS, MCLLSS_CT, and MCLLSS_T have to have the same size");

			return true;
		}

		inline const char* ExtLangmuirLSSParamHandler::identifier() CADET_NOEXCEPT { return "EXT_MULTI_COMPONENT_LANGMUIR_LSS"; }

		inline bool ExtLangmuirLSSParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
		{
			if ((_kkin.size() != _qMax.size()) || (_kkin.size() != _b0.size()) || (_kkin.size() != _ss.size()) || (_kkin.size() != _ct.size()) || (_kkin.size() != _T.size()) || (_kkin.size() < nComp))
				throw InvalidParameterException("EXT_MCLLSS_KKIN, EXT_MCLLSS_QMAX, EXT_MCLLSS_B0, EXT_MCLLSS_SS, EXT_MCLLSS_CT, and EXT_MCLLSS_T have to have the same size");

			return true;
		}


		template <class ParamHandler_t>
		class LangmuirLSSBindingBase : public ParamHandlerBindingModelBase<ParamHandler_t>
		{
		public:

			LangmuirLSSBindingBase() { }
			virtual ~LangmuirLSSBindingBase() CADET_NOEXCEPT { }

			static const char* identifier() { return ParamHandler_t::identifier(); }

			CADET_BINDINGMODELBASE_BOILERPLATE

		protected:
			using ParamHandlerBindingModelBase<ParamHandler_t>::_paramHandler;
			using ParamHandlerBindingModelBase<ParamHandler_t>::_reactionQuasistationarity;
			using ParamHandlerBindingModelBase<ParamHandler_t>::_nComp;
			using ParamHandlerBindingModelBase<ParamHandler_t>::_nBoundStates;

			virtual bool implementsAnalyticJacobian() const CADET_NOEXCEPT { return true; }

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

					const double kkin = static_cast<double>(p->kkin[i]);
					const double qMax = static_cast<double>(p->qMax[i]);
					const double b0 = static_cast<double>(p->b0[i]);
					const double ss = static_cast<double>(p->ss[i]);
					const double ct = static_cast<double>(p->ct[i]);
					const double T = static_cast<double>(p->T[i]);

					// Residual
					//const double b = b0 * exp(-ss * yCp[0]) * exp(ct * 1000 * (1 / T - 1 / 298.15));
					//res[bndIdx] = kkin * (y[bndIdx] - (qMax * b * yCp[i]) / (1 + b * yCp[i]));
					res[bndIdx] = kkin * (y[bndIdx] - (qMax * b0 * exp(-ss * yCp[0]) * exp(ct * 1000 * (1 / T - 1 / 298.15)) * yCp[i]) / (1 + b0 * exp(-ss * yCp[0]) * exp(ct * 1000 * (1 / T - 1 / 298.15)) * yCp[i]));

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

					const double kkin = static_cast<double>(p->kkin[i]);
					const double qMax = static_cast<double>(p->qMax[i]);
					const double b0 = static_cast<double>(p->b0[i]);
					const double ss = static_cast<double>(p->ss[i]);
					const double ct = static_cast<double>(p->ct[i]);
					const double T = static_cast<double>(p->T[i]);

					// Add to dres_i / dq_i
					jac[0] += kkin;

					// Add to dres_i / dc_i
					//double b = b0 * exp(-ss * yCp[0]) * exp(ct * 1000 * (1 / T - 1 / 298.15));
					jac[i - bndIdx - offsetCp] = -kkin * qMax * b0 * exp(-ss * yCp[0]) * exp(ct * 1000 * (1 / T - 1 / 298.15)) / ((1 + b0 * exp(-ss * yCp[0]) * exp(ct * 1000 * (1 / T - 1 / 298.15)) * yCp[i]) * (1 + b0 * exp(-ss * yCp[0]) * exp(ct * 1000 * (1 / T - 1 / 298.15)) * yCp[i]));

					// Advance to next flux and Jacobian row
					++bndIdx;
					++jac;
				}
			}
		};

		typedef LangmuirLSSBindingBase<LangmuirLSSParamHandler> LangmuirLSSBinding;
		typedef LangmuirLSSBindingBase<ExtLangmuirLSSParamHandler> ExternalLangmuirLSSBinding;

		namespace binding
		{
			void registerLangmuirLSSModel(std::unordered_map<std::string, std::function<model::IBindingModel* ()>>& bindings)
			{
				bindings[LangmuirLSSBinding::identifier()] = []() { return new LangmuirLSSBinding(); };
				bindings[ExternalLangmuirLSSBinding::identifier()] = []() { return new ExternalLangmuirLSSBinding(); };
			}
		}  // namespace binding

	}  // namespace model

}  // namespace cadet
