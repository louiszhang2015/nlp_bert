# ======================================================================================================================
# COPYRIGHT (C) WLUPER LTD. - ALL RIGHTS RESERVED 2017 - Present
#
# UNAUTHORIZED COPYING, USE, REPRODUCTION OR DISTRIBUTION OF THIS FILE, VIA ANY MEDIUM IS STRICTLY PROHIBITED.
# ALL CONTENTS ARE PROPRIETARY AND CONFIDENTIAL.
#
# WRITTEN BY:
#   Wluper Tech Team <nikolai@wluper.com>
#
# GENERAL ENQUIRIES:
#   <contact@wluper.com>
# ======================================================================================================================

# ======================================================================================================================
#
# IMPORTS
#
# ======================================================================================================================

from setuptools import find_packages, setup

# ======================================================================================================================



setup(
    # =====
    # Setup
    # =====
    name="bert_exp",
    version="0.0.0.0",
    description="bert_exp",
    url="",
    author="bingbing zhang",
    author_email="zbb920508@gmail.com",
    license="WLUPER",
    # =================================
    # Actual packages, data and scripts
    # =================================
    packages=find_packages(),
    package_dir={"bert_exp": "bert_exp"},
    package_data={
        "bert_exp": [
            # If you want to add data to the package, put the path to it relative to the package here
        ]
    },
    install_requires=[
    ],
    extras_require={
        "dev": [
            "freezegun==1.1.0",
            "bump2version==1.0.1",
            "black==21.7b0",
            "isort==5.9.3",
            "flake8==3.9.2",
            "pre-commit==2.14.0",
        ],
    },
    python_requires=">=3.6",
)
