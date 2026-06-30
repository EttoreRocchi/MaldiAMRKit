Dataset Builder & Loader
========================

Build and load datasets using pluggable layout adapters.

DatasetBuilder
--------------

.. autoclass:: maldiamrkit.data.DatasetBuilder
   :members:
   :undoc-members:

DatasetLoader
-------------

.. autoclass:: maldiamrkit.data.DatasetLoader
   :members:
   :undoc-members:

Input Layouts
-------------

.. autoclass:: maldiamrkit.data.InputLayout
   :members:

.. autoclass:: maldiamrkit.data.FlatLayout
   :members:
   :undoc-members:

.. autoclass:: maldiamrkit.data.BrukerTreeLayout
   :members:
   :undoc-members:

Dataset Layouts
---------------

.. autoclass:: maldiamrkit.data.DatasetLayout
   :members:

.. autoclass:: maldiamrkit.data.DRIAMSLayout
   :members:
   :undoc-members:

.. autoclass:: maldiamrkit.data.MARISMaLayout
   :members:
   :undoc-members:

.. autofunction:: maldiamrkit.data.strip_driams_replicate

ProcessingHandler
-----------------

.. autoclass:: maldiamrkit.data.ProcessingHandler
   :members:
   :undoc-members:

BuildReport
-----------

.. autoclass:: maldiamrkit.data.BuildReport
   :no-members:

Duplicate Handling
------------------

.. autoclass:: maldiamrkit.data.DuplicateStrategy
   :members:
   :undoc-members:
   :show-inheritance:

Dataset Manifest
----------------

Every dataset produced by :class:`~maldiamrkit.data.DatasetBuilder` carries a
self-describing ``site_info.json`` manifest at its root, so it can be re-opened
without external knowledge. Downstream layouts (notably
:class:`~maldiamrkit.data.DRIAMSLayout`) consult it at load time to pre-fill
unspecified constructor kwargs.

.. autoclass:: maldiamrkit.data.SiteInfo
   :members:

.. autoclass:: maldiamrkit.data.BuildInfo
   :members:

.. autofunction:: maldiamrkit.data.read_site_info

.. autofunction:: maldiamrkit.data.write_site_info
