from arocr.cards import _clean_address, _clean_name
from arocr.pipeline import best_national_id, date_prefix, valid_national_id


def test_national_id():
    assert valid_national_id("29001010112341")
    assert not valid_national_id("2900101011234")       # 13 digits
    assert not valid_national_id("29013010112345")      # month 13
    assert not valid_national_id("29001019912345")      # governorate 99
    assert not valid_national_id("29001010112441")      # one digit misread: check digit fails


def test_check_digit_picks_reading():
    # the same number read several ways; the majority "2" is wrong, the check digit knows
    reads = ["29001010112941", "29001010112241", "29001010112241", "29001010112341", "29001010112341"]
    assert best_national_id(reads) == "29001010112341"
    assert best_national_id(["29001010112241", "29001010112241"]) == ""        # agreeing on a wrong digit is not enough
    # one_digit_only: readings differ in one place, one of them fits
    assert best_national_id(["29001010112941", "29001010112241", "29001010112341"]) == "29001010112341"


def test_clean_name():
    assert _clean_name("عبد انمنعم احمد") == "عبدالمنعم احمد"
    assert _clean_name("يو سف عمر") == "يوسف عمر"
    assert _clean_name("محم1 حمدى") == "محم حمدى"
    assert _clean_name("رجب انتور مجمود") == "رجب انور محمود"
    assert _clean_name("سيد فولى حسن") == "سيد فولى حسن"      # a real name, not "فوزى"


def test_clean_address():
    assert _clean_address("مركز أبوقرقاص - انمنيا") == "مركز أبوقرقاص - المنيا"
    assert _clean_address("شارع الابطال") == "شارع الابطال"


def test_date_checks_id():
    # the birth date on the card is the ID's first 7 digits
    assert date_prefix("1990/01/01") == "2900101"
    assert date_prefix("1990/13/01") == ""
    # the date part of the number misread twice ("2909..."): the date line repairs it
    bad = "29091010112341"
    assert best_national_id([bad, bad], "1990/01/01") == "29001010112341"
    assert best_national_id([bad, bad]) == ""
