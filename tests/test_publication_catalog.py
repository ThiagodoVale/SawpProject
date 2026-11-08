from sawp.publication.catalog import PUBLICATION_PRODUCTS


def test_publication_catalog_has_all_18_unique_products() -> None:
    identifiers = [product.product_id for product in PUBLICATION_PRODUCTS]
    filenames = [product.filename for product in PUBLICATION_PRODUCTS]
    assert len(identifiers) == 18
    assert len(set(identifiers)) == 18
    assert len(set(filenames)) == 18
    assert filenames == [
        *(f"Figure{number}.png" for number in range(1, 10)),
        *(f"FigureS{number}.png" for number in range(1, 10)),
    ]
