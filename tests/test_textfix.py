from arocr.textfix import visual_to_logical


def test_pure_arabic_is_mirrored_back():
    # the case from PdfOcrApp: model output was the exact reverse
    assert visual_to_logical("رصنلا فيس تحدم رمع") == "عمر مدحت سيف النصر"


def test_number_inside_arabic_keeps_its_order():
    # "عمارة ١٢" — reversing the whole line would give ٢١
    assert visual_to_logical("١٢ ةرامع") == "عمارة ١٢"
    assert visual_to_logical("12 ةرامع") == "عمارة 12"


def test_date_and_id_inside_arabic():
    assert visual_to_logical("١٩٩٠/٠٥/٠٣ داليملا خيرات") == "تاريخ الميلاد ١٩٩٠/٠٥/٠٣"
    assert visual_to_logical("29001010112345 يموقلا مقرلا") == "الرقم القومي 29001010112345"


def test_latin_code_inside_arabic():
    assert visual_to_logical("KM8424714 مقر") == "رقم KM8424714"


def test_brackets_are_mirrored():
    # visually the opening bracket sits on the right, drawn as ")"
    assert visual_to_logical("(رصم)") == "(مصر)"


def test_lines_without_arabic_are_untouched():
    assert visual_to_logical("٢٩٠٠١٠١٠١١٢٣٤٥") == "٢٩٠٠١٠١٠١١٢٣٤٥"
    assert visual_to_logical("Invoice 2026/10") == "Invoice 2026/10"


def test_persian_forms_fold_to_arabic():
    from arocr.textfix import normalize_egyptian
    assert normalize_egyptian("کمبیوتر ۱۲") == "كمبيوتر ١٢"


def test_wrapped_reference_tail():
    # "(رو ٥: ٦)." wrapped so the second line has no Arabic letters
    assert visual_to_logical(".(٦ :٥") == "٥: ٦)."


def test_number_line_keeps_group_order():
    # national ID printed in spaced groups: a number, not RTL text
    assert visual_to_logical("٢٩٨ ٠٧ ١٤") == "٢٩٨ ٠٧ ١٤"
    assert visual_to_logical("١٩٩٨/٠٧") == "١٩٩٨/٠٧"


def test_card_codes():
    from arocr.textfix import fix_codes
    assert fix_codes("LEO976591") == "LE0976591"
    assert fix_codes("HS6709874") == "HS6709874"
    assert fix_codes("OPEN") == "OPEN"


def test_compound_names():
    from arocr.textfix import fix_compound_names
    assert fix_compound_names("خميس عبدائر حيم محمد") == "خميس عبدالرحيم محمد"
    assert fix_compound_names("مركز أبوقر قاص - المنيا") == "مركز أبوقرقاص - المنيا"
    assert fix_compound_names("محمد عبد الله") == "محمد عبد الله"      # written spaced: leave it


def test_known_phrase():
    from arocr.textfix import match_phrase
    assert match_phrase("بطاقة تحقيق الشخمية") == "بطاقة تحقيق الشخصية"
    assert match_phrase("ذهب الولد إلى المدرسة") is None


def test_tanween():
    from arocr.textfix import fix_tanween
    assert fix_tanween("متواترأً يومأ بعد يوم") == "متواتراً يوماً بعد يوم"
    assert fix_tanween("شيئأ أصلأ.") == "شيئاً أصلاً."
    assert fix_tanween("بدأ يقرأ الخطأ وبدأ") == "بدأ يقرأ الخطأ وبدأ"     # real final hamza stays


def test_book_reader_fixes():
    from arocr.textfix import fix_allah, fix_shadda_tanween
    assert fix_allah("بواسطة اله نفسه باله وإله") == "بواسطة الله نفسه بالله وإله"
    assert fix_shadda_tanween("شاهدّا أمّا") == "شاهداً أمّا"
