from utils import impute
from utils.model.novel import NOVEL_MODEL_VARS


def test_determine_imputations(simple_df_with_missingness_fixture):
    n_imps, fraction = impute.determine_n_imputations(
        simple_df_with_missingness_fixture
    )
    assert (n_imps, fraction) == (40, 0.4)


# def test_splitter_winsor_mice(initial_df_fixture, train_test_split_fixture):
#     # TODO: Finish this test
#     raise NotImplementedError
#     swm = impute.SplitterWinsorMICE(
#         df=input_df_fixture,
#         train_test_splitter=train_test_split_fixture,
#         target_variable_name=NOVEL_MODEL_VARS["target"],
#         cont_variables=NOVEL_MODEL_VARS["cont"],
#         binary_variables=NOVEL_MODEL_VARS["cat"],
#         winsor_quantiles=(0.2, 0.8),
#         winsor_include={
#             "S01AgeOnArrival": (False, True),
#             "S03GlasgowComaScore": (False, False),
#         },
#         n_mice_imputations=5,
#         n_mice_burn_in=10,
#         n_mice_skip=3,
#     )
#     swm.split_winsorize_mice()
