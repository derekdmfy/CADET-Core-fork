function [purityRequirement, failureProb, lowerQuant, upperQuant] = riskAnalysisPostProcessRep(purities, quantiles)
%RISKANALYSISPOSTPROCESSREP Computes failure probabilities from purity samples
%
%   The probability of failing a purity requirement is analyzed by a Monte
%   Carlo simulation. It is assumed that the loading concentrations are
%   subject to noise following a multivariate Gaussian distribution. By first
%   sampling loading concentrations from this distribution and computing
%   purity (and yield) for each sample, purity samples are created. In a
%   second step the failure probability is evaluated by computing a mean and
%   the variation of the probability is assessed by analyzing repetitions.
%   The cut points are fixed in this study and the the noise on each
%   component is assumed to be uncorrelated with given standard deviation.
%
%   This function performs the failure probability calculation and computes
%   bounds on each probability (e.g., via confidence intervals). In order
%   to do the latter, the given input data is partitioned into 10 slices that
%   can be considered repetitions of the original experiment. Depending on the
%   QUANTILES switch, either minimum and maximum value for each probability
%   are reported, or the repetitions are used to form a discrete distribution
%   for each probability and 95 % confidence bounds are computed from its
%   quantiles. See RISKANALYSISSAMPLING() for sample generation.
%
%   The model describes ion-exchange chromatography of lysozyme, cytochrome,
%   and ribonuclease on the strong cation-exchanger SP Sepharose FF. Model
%   parameters are taken from benchmark 2 of the following publication:
%   A. Püttmann, S. Schnittert, U. Naumann & E. von Lieres (2013).
%   Fast and accurate parameter sensitivities for the general rate model of
%   column liquid chromatography.
%   Computers & Chemical Engineering, 56, 46–57.
%   doi:10.1016/j.compchemeng.2013.04.021
%
%   RISKANALYSISPOSTPROCESS(PURITIES, QUANTILES) calculates failure
%   probabilities using the purity samples generated by RISKANALYSISSAMPLING.
%   Also returns minimum and maximum failure probabilities from repetitions if
%   QUANTILES = false and 95 % confidence intervals if QUANTILES = true.
%
%   [PURITYREQUIREMENT, FAILUREPROB, LOWERQUANT, UPPERQUANT] = RISKANALYSISPOSTPROCESS(...)
%   returns a vector PURITYREQUIREMENT with ascending purity requirements,
%   the corresponding failure probabilities of not meeting the requirement
%   in FAILUREPROB, lower bounds of the 95 % confidence band on the
%   failure probabilities in LOWERQUANT, and the corresponding upper bounds
%   in the vector UPPERQUANT.
%
%   See also RISKANALYSISSAMPLING, RISKANALYSISPOSTPROCESS

% Copyright: (C) 2008-2017 The CADET Authors
%            See the license note at the end of the file.

	% Number of repetitions
	nRep = 10;
	% Confidence level (95 %)
	conf = 0.95;

	% Calculate failure probabilities from samples for varying purity
	% requirements
	purityRequirement = linspace(0.9, 1, 201)';

	% Repetitions
	nRepSamples = length(purities) / nRep;
	failureProbSamples = zeros(length(purityRequirement), nRep);
	for i = 1:nRep
		% Generate new purity samples by randomly drawing with replacement
		
		newPur = purities((i-1)*nRepSamples+1:i*nRepSamples);
		
		% Again, compute failure probabilities for varying purity levels
		for j = 1:length(purityRequirement)
			failureProbSamples(j, i) = sum(newPur < purityRequirement(j)) / length(newPur);
		end
	end
	
	% Compute quantiles of failure probability samples to obtain confidence
	% band
	if quantiles
		lowerQuant = zeros(length(purityRequirement), 1);
		upperQuant = zeros(length(purityRequirement), 1);
		for i = 1:length(purityRequirement)
			d = sort(failureProbSamples(i, :));

			quant = [0 (0.5:(nRep-0.5))./nRep 1]';
			vals = [d(1); d(:); d(end)];
			temp = interp1(quant, vals, ((1-conf) / 2) + [0, conf]);
			lowerQuant(i) = temp(1);
			upperQuant(i) = temp(2);
		end
	else
		lowerQuant = min(failureProbSamples, [], 2);
		upperQuant = max(failureProbSamples, [], 2);
	end
	failureProb = mean(failureProbSamples, 2);
	
	% Plot results with confidence band as error bars
	errorbar(purityRequirement, failureProb, failureProb-lowerQuant, upperQuant-failureProb);
	grid on;
	xlabel('Purity requirement beta');
	ylabel('Failure probability');
	xlim([min(purityRequirement), max(purityRequirement)]);
	ylim([min(failureProb), max(failureProb)]);
end

% =============================================================================
%  CADET
%  
%  Copyright (C) 2008-2017: The CADET Authors
%            Please see the AUTHORS and CONTRIBUTORS file.
%  
%  All rights reserved. This program and the accompanying materials
%  are made available under the terms of the GNU Public License v3.0 (or, at
%  your option, any later version) which accompanies this distribution, and
%  is available at http://www.gnu.org/licenses/gpl.html
% =============================================================================
