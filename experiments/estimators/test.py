from .bbsd import Estimator as BBSDEstimator
from ..datasets import pdtb, biodrb, ted
from .error import Estimator as Error
from .erm import LinearHypothesisSpace
from .erm import PyTorchEstimator as HypothesisEstimator
from .hypothesis_divergence import Estimator \
    as HypothesisDivergence
from .hypothesis_class_divergence import Estimator \
    as HypothesisClassDivergence
from .two_sample import Estimator as TwoSampleEstimator

def test_hypothesis_divergences():
    raise Exception('Outdated Test...')
    pdtb_train = pdtb(train=True)
    pdtb_test = pdtb(train=False)
    biodrb_train = biodrb(train=True)
    hypothesis_space = LinearHypothesisSpace(
        num_classes=pdtb_train.num_classes,
        num_inputs=pdtb_train.input_sz)
    print('Training hypothesis...')
    hypothesis = HypothesisEstimator(hypothesis_space, 
        pdtb_train, verbose=True).compute()
    results = dict()
    print('Getting train error...')
    results['train_error'] = Error(hypothesis, hypothesis_space,
        pdtb_train, verbose=True).compute()
    print('Getting transfer error on pdtb...')
    results['pdtb_test_error'] = Error(hypothesis, 
        hypothesis_space, pdtb_test, verbose=True).compute()
    print('Getting transfer error on biodrb...')
    results['biodrb_test_error'] = Error(hypothesis, 
        hypothesis_space, biodrb_train, verbose=True).compute()
    print('Getting bbds against pdtb')
    results[f'bbsd_pdtb'] = BBSDEstimator(hypothesis, 
        hypothesis_space, pdtb_train, pdtb_test).compute()
    print('Getting bbds against biodrb')
    results[f'bbsd_biodrb'] = BBSDEstimator(hypothesis, 
        hypothesis_space, pdtb_train, pdtb_test).compute()
    print('Getting h_divergence against pdtb...')
    results['h_divergence_pdtb'] = HypothesisDivergence(
        hypothesis, hypothesis_space, pdtb_train, pdtb_test,
        verbose=True).compute()
    print('Getting h_divergence against biodrb...')
    results['h_divergence_biodrb'] = HypothesisDivergence(
        hypothesis, hypothesis_space, pdtb_train, biodrb_train,
        verbose=True).compute()
    print('Getting H_divergence against pdtb...')
    results['H_divergence_pdtb'] = HypothesisClassDivergence(
        hypothesis_space, pdtb_train, pdtb_test,
        verbose=True).compute()
    print('Getting H_divergence against biodrb...')
    results['H_divergence_biodrb'] = HypothesisClassDivergence(
        hypothesis_space, pdtb_train, biodrb_train,
        verbose=True).compute()
    return results

def test_twosample_statistics():
    raise Exception('Outdated test')
    pdtb_train = pdtb(train=True)
    pdtb_test = pdtb(train=False)
    biodrb_train = biodrb(train=True)
    results = dict()
    dsets = [('pdtb', pdtb_test), ('biodrb', biodrb_train)]
    for desc, dset in dsets:
        for stat in TwoSampleEstimator.STATS:
            print(f'Getting {stat} for {desc}...')
            results[f'{desc}_{stat}'] = TwoSampleEstimator(stat, 
                pdtb_train, dset).compute()
    return results

def add_dicts(a, b):
    for k, v in b.items():
        if k not in a:
            a[k] = v
    return a

def test_total_variation_bbsd():
    pdtb_train = pdtb(train=False)
    pdtb_test = pdtb(train=True)
    biodrb_train = biodrb(train=False)
    ted_train = ted(train=False)
    hypothesis_space = LinearHypothesisSpace(
        num_classes=pdtb_train.num_classes,
        num_inputs=pdtb_train.input_sz)
    print('Training hypothesis...')
    hypothesis = HypothesisEstimator(hypothesis_space, 
        pdtb_train, verbose=True).compute()
    print('pdtb v. pdtb:',
        BBSDEstimator(None, hypothesis_space,
            pdtb_train, pdtb_test, 'tv', 
            device='cpu', verbose=True).compute())
    print('pdtb v. biodrb:',
        BBSDEstimator(None, hypothesis_space,
            pdtb_train, biodrb_train, 'tv', 
            device='cpu', verbose=True).compute())
    print('pdtb v. ted:',
        BBSDEstimator(None, hypothesis_space,
            pdtb_train, ted_train, 'tv', 
            device='cpu', verbose=True).compute())
    print('two sample mmd against biodrb:',
        TwoSampleEstimator('mmd', pdtb_train, 
        biodrb_train).compute())

if __name__ == '__main__':
    # OUTDATED TESTS
    # results = test_hypothesis_divergences()
    # results = add_dicts(results, test_twosample_statistics())
    # for k, v in results.items():
    #     print(k, ':', f'{v:.4f}')
    test_total_variation_bbsd()
