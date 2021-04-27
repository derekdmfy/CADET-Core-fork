// =============================================================================
//  CADET
//  
//  Copyright © 2008-2021: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

#include "model/binding/BindingModelBase.hpp"
#include "model/ExternalFunctionSupport.hpp"
#include "model/binding/BindingModelMacros.hpp"
#include "model/binding/RefConcentrationSupport.hpp"
#include "model/ModelUtils.hpp"
#include "cadet/Exceptions.hpp"
#include "model/Parameters.hpp"
#include "LocalVector.hpp"
#include "SimulationTypes.hpp"

#include <functional>
#include <unordered_map>
#include <string>
#include <vector>

/*<codegen>
{
	"name": "SelfAssociationParamHandler",
	"externalName": "ExtSelfAssociationParamHandler",
	"parameters":
		[
			{ "type": "ScalarParameter", "varName": "lambda", "confName": "SAI_LAMBDA"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kA", "confName": "SAI_KA1"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kA2", "confName": "SAI_KA2"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kD", "confName": "SAI_KD"},
			{ "type": "ScalarComponentDependentParameter", "varName": "nu", "confName": "SAI_NU"},
			{ "type": "ScalarComponentDependentParameter", "varName": "sigma", "confName": "SAI_SIGMA"}
		],
	"constantParameters":
		[
			{ "type": "ReferenceConcentrationParameter", "varName": ["refC0", "refQ"], "objName": "refConcentration", "confPrefix": "SAI_"}
		]
}
</codegen>*/

/* Parameter description
 ------------------------
 lambda = Ionic capacity
 kA = Adsorption rate
 kA2 = Adsorption rate of dimerization
 kD = Desorption rate
 nu = Characteristic charge
 sigma = Steric factor
 refC0, refQ = Reference concentrations
*/

namespace cadet
{

namespace model
{

inline const char* SelfAssociationParamHandler::identifier() CADET_NOEXCEPT { return "SELF_ASSOCIATION"; }

inline bool SelfAssociationParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	if ((_kA.size() != _kA2.size()) || (_kA.size() != _kD.size()) || (_kA.size() != _nu.size()) 
		|| (_kA.size() != _sigma.size()) || (_kA.size() < nComp))
		throw InvalidParameterException("SAI_KA1, SAI_KA2, SAI_KD, SAI_NU, and SAI_SIGMA have to have the same size");

	// Assume monovalent salt ions by default
	if (_nu.get()[0] <= 0.0)
		_nu.get()[0] = 1.0;

	return true;
}

inline const char* ExtSelfAssociationParamHandler::identifier() CADET_NOEXCEPT { return "EXT_SELF_ASSOCIATION"; }

inline bool ExtSelfAssociationParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	if ((_kA.size() != _kA2.size()) || (_kA.size() != _kD.size()) || (_kA.size() != _nu.size()) 
		|| (_kA.size() != _sigma.size()) || (_kA.size() < nComp))
		throw InvalidParameterException("EXT_SAI_KA1, EXT_SAI_KA2, EXT_SAI_KD, EXT_SAI_NU, and EXT_SAI_SIGMA have to have the same size");

	// Assume monovalent salt ions by default
	if (_nu.base()[0] <= 0.0)
		_nu.base()[0] = 1.0;

	return true;
}


/**
 * @brief Defines the self association binding model
 * @details Implements the self association adsorption model: \f[ \begin{align} 
 *              q_0 &= \Lambda - \sum_{j} \nu_j q_j \\
 *              \frac{\mathrm{d}q_i}{\mathrm{d}t} &= \left[k_{a1,i} + k_{a2,i} c_{p,i} \right] c_{p,i} \left( \Lambda - \sum_j\left( \nu_j + \sigma_j \right) q_j \right)^{\nu_i} - k_{d,i} q_i c_{p,0}^{\nu_i} 
 *          \end{align} \f]
 *          Here, the coefficient @f$ k_{a2,i} @f$ denotes the adsorption rate of dimerization.
 *          Component @c 0 is assumed to be salt. Multiple bound states are not supported. 
 *          Components without bound state (i.e., non-binding components) are supported.
 *          
 *          See @cite Mollerup2008 and @cite Westerberg2012.
 * @tparam ParamHandler_t Type that can add support for external function dependence
 */
template <class ParamHandler_t>
class SelfAssociationBindingBase : public ParamHandlerBindingModelBase<ParamHandler_t>
{
public:

	SelfAssociationBindingBase() { }
	virtual ~SelfAssociationBindingBase() CADET_NOEXCEPT { }

	static const char* identifier() { return ParamHandler_t::identifier(); }

	virtual bool configureModelDiscretization(IParameterProvider& paramProvider, unsigned int nComp, unsigned int const* nBound, unsigned int const* boundOffset)
	{
		const bool res = BindingModelBase::configureModelDiscretization(paramProvider, nComp, nBound, boundOffset);

		// Guarantee that salt has exactly one bound state
		if (nBound[0] != 1)
			throw InvalidParameterException("Self association binding model requires exactly one bound state for salt component");

		// First flux is salt, which is always quasi-stationary
		_reactionQuasistationarity[0] = true;

		return res;
	}

	virtual bool hasSalt() const CADET_NOEXCEPT { return true; }
	virtual bool supportsMultistate() const CADET_NOEXCEPT { return false; }
	virtual bool supportsNonBinding() const CADET_NOEXCEPT { return true; }
	virtual bool hasQuasiStationaryReactions() const CADET_NOEXCEPT { return true; }
	virtual bool implementsAnalyticJacobian() const CADET_NOEXCEPT { return true; }

	virtual bool preConsistentInitialState(double t, unsigned int secIdx, const ColumnPosition& colPos, double* y, double const* yCp, LinearBufferAllocator workSpace) const
	{
		typename ParamHandler_t::ParamsHandle const p = _paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		// Compute salt component from given bound states q_j

		// Salt equation: nu_0 * q_0 - Lambda + Sum[nu_j * q_j, j] == 0
		//           <=>  q_0 == (Lambda - Sum[nu_j * q_j, j]) / nu_0
		y[0] = static_cast<double>(p->lambda);

		unsigned int bndIdx = 1;
		for (int j = 1; j < _nComp; ++j)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[j] == 0)
				continue;

			y[0] -= static_cast<double>(p->nu[j]) * y[bndIdx];

			// Next bound component
			++bndIdx;
		}

		y[0] /= static_cast<double>(p->nu[0]);

		return true;
	}
	
	virtual void postConsistentInitialState(double t, unsigned int secIdx, const ColumnPosition& colPos, double* y, double const* yCp, LinearBufferAllocator workSpace) const
	{
		preConsistentInitialState(t, secIdx, colPos, y, yCp, workSpace);
	}

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
		using CpStateParamType = typename DoubleActivePromoter<CpStateType, ParamType>::type;
		using StateParamType = typename DoubleActivePromoter<StateType, ParamType>::type;

		typename ParamHandler_t::ParamsHandle const p = _paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		// Salt flux: nu_0 * q_0 - Lambda + Sum[nu_j * q_j, j] == 0 
		//       <=>  nu_0 * q_0 == Lambda - Sum[nu_j * q_j, j] 
		// Also compute \bar{q}_0 = nu_0 * q_0 - Sum[sigma_j * q_j, j]
		res[0] = static_cast<ParamType>(p->nu[0]) * y[0] - static_cast<ParamType>(p->lambda);
		StateParamType q0_bar = static_cast<ParamType>(p->nu[0]) * y[0];

		unsigned int bndIdx = 1;
		for (int j = 1; j < _nComp; ++j)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[j] == 0)
				continue;

			res[0] += static_cast<ParamType>(p->nu[j]) * y[bndIdx];
			q0_bar -= static_cast<ParamType>(p->sigma[j]) * y[bndIdx];

			// Next bound component
			++bndIdx;
		}

		const StateParamType q0_bar_divRef = q0_bar / static_cast<ParamType>(p->refQ);
		const CpStateParamType yCp0_divRef = yCp[0] / static_cast<ParamType>(p->refC0);

		// Protein fluxes: -(k_{a,i} + k_{a2,i} * c_{p,i}) * c_{p,i} * \bar{q}_0^{nu_i / nu_0} + k_{d,i} * q_i * c_{p,0}^{nu_i / nu_0}
		bndIdx = 1;
		for (int i = 1; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			const ParamType nu_over_nu0 = static_cast<ParamType>(p->nu[i]) / static_cast<ParamType>(p->nu[0]);
			const CpStateParamType c0_pow_nu = pow(yCp0_divRef, nu_over_nu0);
			const StateParamType q0_bar_pow_nu = pow(q0_bar_divRef, nu_over_nu0);

			// Residual
			res[bndIdx] = static_cast<ParamType>(p->kD[i]) * y[bndIdx] * c0_pow_nu - yCp[i] * (static_cast<ParamType>(p->kA[i]) + yCp[i] * static_cast<ParamType>(p->kA2[i])) * q0_bar_pow_nu;

			// Next bound component
			++bndIdx;
		}

		return 0;
	}

	template <typename RowIterator>
	void jacobianImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double const* yCp, int offsetCp, RowIterator jac, LinearBufferAllocator workSpace) const
	{
		typename ParamHandler_t::ParamsHandle const p = _paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		double q0_bar = static_cast<double>(p->nu[0]) * y[0];

		// Salt flux: nu_0 * q_0 - Lambda + Sum[nu_j * q_j, j] == 0
		jac[0] = static_cast<double>(p->nu[0]);
		int bndIdx = 1;
		for (int j = 1; j < _nComp; ++j)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[j] == 0)
				continue;

			jac[bndIdx] = static_cast<double>(p->nu[j]);

			// Calculate \bar{q}_0 = nu_0 * q_0 - Sum[sigma_j * q_j, j]
			q0_bar -= static_cast<double>(p->sigma[j]) * y[bndIdx];

			// Next bound component
			++bndIdx;
		}

		// Advance to protein fluxes
		++jac;

		const double refC0 = static_cast<double>(p->refC0);
		const double refQ = static_cast<double>(p->refQ);
		const double yCp0_divRef = yCp[0] / refC0;
		const double q0_bar_divRef = q0_bar / refQ;

		// Protein fluxes: -(k_{a,i} + k_{a2,i} * c_{p,i}) * c_{p,i} * \bar{q}_0^{nu_i / nu_0} + k_{d,i} * q_i * c_{p,0}^{nu_i / nu_0}
		// We have already computed \bar{q}_0 in the loop above
		bndIdx = 1;
		for (int i = 1; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			// Getting to c_{p,0}: -bndIdx takes us to q_0, another -offsetCp to c_{p,0}. This means jac[-bndIdx - offsetCp] corresponds to c_{p,0}.
			// Getting to c_{p,i}: -bndIdx takes us to q_0, another -offsetCp to c_{p,0} and a +i to c_{p,i}.
			//                     This means jac[i - bndIdx - offsetCp] corresponds to c_{p,i}.

			const double ka = static_cast<double>(p->kA[i]);
			const double ka2 = static_cast<double>(p->kA2[i]);
			const double kd = static_cast<double>(p->kD[i]);
			const double nu = static_cast<double>(p->nu[i]) / static_cast<double>(p->nu[0]);

			const double effKa = yCp[i] * (ka + ka2 * yCp[i]);
			const double c0_pow_nu     = pow(yCp0_divRef, nu);
			const double q0_bar_pow_nu = pow(q0_bar_divRef, nu);
			const double c0_pow_nu_m1_divRef     = pow(yCp0_divRef, nu - 1.0) / refC0;
			const double q0_bar_pow_nu_m1_divRef = pow(q0_bar_divRef, nu - 1.0) / refQ;

			// dres_i / dc_{p,0}
			jac[-bndIdx - offsetCp] = kd * y[bndIdx] * nu * c0_pow_nu_m1_divRef;
			// dres_i / dc_{p,i}
			jac[i - bndIdx - offsetCp] = -(ka + 2.0 * ka2 * yCp[i]) * q0_bar_pow_nu;
			// dres_i / dq_0
			jac[-bndIdx] = -effKa * nu * q0_bar_pow_nu_m1_divRef * static_cast<double>(p->nu[0]);

			// Fill dres_i / dq_j
			int bndIdx2 = 1;
			for (int j = 1; j < _nComp; ++j)
			{
				// Skip components without bound states (bound state index bndIdx is not advanced)
				if (_nBoundStates[j] == 0)
					continue;

				// dres_i / dq_j
				jac[bndIdx2 - bndIdx] = -effKa * nu * q0_bar_pow_nu_m1_divRef * (-static_cast<double>(p->sigma[j]));
				// Getting to q_j: -bndIdx takes us to q_0, another +bndIdx2 to q_j. This means jac[bndIdx2 - bndIdx] corresponds to q_j.

				++bndIdx2;
			}

			// Add to dres_i / dq_i
			jac[0] += kd * c0_pow_nu;

			// Advance to next flux and Jacobian row
			++bndIdx;
			++jac;
		}
	}
};


typedef SelfAssociationBindingBase<SelfAssociationParamHandler> SelfAssociationBinding;
typedef SelfAssociationBindingBase<ExtSelfAssociationParamHandler> ExternalSelfAssociationBinding;

namespace binding
{
	void registerSelfAssociationModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings)
	{
		bindings[SelfAssociationBinding::identifier()] = []() { return new SelfAssociationBinding(); };
		bindings[ExternalSelfAssociationBinding::identifier()] = []() { return new ExternalSelfAssociationBinding(); };
	}
}  // namespace binding

}  // namespace model

}  // namespace cadet
