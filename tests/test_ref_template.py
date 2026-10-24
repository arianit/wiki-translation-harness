from wiki_translation_harness.ref_template import expand_r_definitions


def test_r_definition_becomes_a_ref_tag_and_reuse_is_left_alone():
    # Regression (Roman Empire): sqwiki's {{R}} cannot define a reference.
    text = "A.{{R|n=Foo|r={{Cite book |title=T |page=1}}}} B.{{R|Foo}}"
    patched, converted, skipped = expand_r_definitions(text)
    assert patched == 'A.<ref name="Foo">{{Cite book |title=T |page=1}}</ref> B.{{R|Foo}}'
    assert (converted, skipped) == (1, 0)


def test_r_definition_keeps_group_and_page():
    patched, _, _ = expand_r_definitions("{{r|Foo|r=Text|g=note|p=5}}")
    assert patched == '<ref name="Foo" group="note">Text</ref>{{Rp|5}}'


def test_r_calls_it_cannot_model_are_left_alone():
    text = "{{R|a|b|r=Text}} {{R|Foo|r=Text|q=quote}} <ref>{{R|n=Z|r=Y}}</ref>"
    assert expand_r_definitions(text) == (text, 0, 3)
