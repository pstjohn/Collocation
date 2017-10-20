import pytest
from collocation.VariableHandler import VariableHandler


variables = {
    'x'  : (20, 10, 5),
    'p'  : (30,),
    'u'  : (40, 20),
}

@pytest.fixture
def variable_handler():
    return VariableHandler(variables)


@pytest.mark.parametrize("suffix", ['_lb', '_ub', '_in', '_op', '_sx'])
def test_sizes(variable_handler, suffix):

    for variable, shape in variables.items():
        varh = getattr(variable_handler, variable + suffix)
        assert varh.shape == shape


@pytest.mark.parametrize("suffix", ['_lb', '_ub', '_in', '_op'])
def test_setter_and_getter(variable_handler, suffix):

    for variable, shape in variables.items():
        varh = getattr(variable_handler, variable + suffix)
        assert varh.shape == shape



