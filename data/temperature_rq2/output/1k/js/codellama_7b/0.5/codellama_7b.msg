Update track color property in trackCore.js to accept an optional parameter for customizing the track color.
Refactored _cleanup method to avoid unnecessary call to _processNext when no more tasks are available in the queue.
Added button type 'Transparent' for begin and end buttons in ResponsivePopover
Update hand-controls component to use LoopRepeat animation loop
Added constants for O_TRUNC, O_CREAT, O_RDWR, O_EXCL, and O_RDONLY from Node.js in rollup.iife.config.js
Fix bug in ${reportName} report with end date before start date. Add error handling.
Added support for Node.js exit code handling on Windows
Fix error handling in afterAll hook for generic role deletion
Update third-party library mapping in test environment
Added support for template error handling in webfont task. (#<HASH>)
Refactor: Improve performance of findModelNodes by using a constant-time lookup for arrays and reducing redundant code.
Update jquery.dirtyforms.js to support IE8 and IE9 input events.
Update AccelerometerProxy for Windows 8 platform
Update normalizeEntrypoints function to support Webpack 4
Added error handling for missing keys in IFrameApi#deriveAddresses()
Update require path for ui-layout component
Added a reference to the validation messages in the `_VALIDATIONS_REF` array for faster access.
Update CLI configuration file for new OS support
Added support for uploading files with a unique ID for each file and directory
Update peity library to latest version
Fix issue with iframe not reloading after template change.
Fix incorrect callSite reference in asyncFunctionInitialized() method to use original callSite instead of slice(2).
Update modal animation type to 'fade'
Fix issue with adding old Microsoft filter for gradient background in IE8. Fixes #1234.
Added support for enter key press on select field to save form data
Update datatable.js to support new sort functionality in PrimeFaces 10
Added data-id attribute to regions for easier identification and styling.
Added support for custom command options
Update IconButton component to support SvgIcon as a child component
Fix issue with transformPosition() method when source is Emitter.sources.USER
Refactored clean.js to use new SelectorsOptimizer class. Removed unused options and added compatibility with aggressive merging.
Update TextMetrics class to support overwriting of text-related properties for multiline styled text.
Update live tab selection on pointer move and up events
Added support for stubbing file system exists check in unit tests
Refactor push payload validation to use `Object.prototype.toString()`
Added support for searching agents by UUID or address in data.js
Update network update frequency in script.js
Fix incorrect namespace handling in executeScript function. Changed behavior of third argument to false for consistency with other functions.
Update current running test export in PlaybackState store
Update script for running webpack with production config file
Update watch options for Windows platform
Update `addComponents` to `addUtilities` in `src/index.js`
Refactored GameEngine class to use new physics engine initialization API
Update queue configuration for BullMQ to include new parameters for Redis connection
Update options for util.inspect() function in index.js file. Add support for deep cloning of objects using extend(true, obj1, obj2) and remove appendTime option.
Update bundleWebpack options for production build
Fast reconnect feature for IRC client, using servers from Servers module.
Update RangeSlider to fire liveChange event when dragging handle
Update custom marker directive to use a new setCustomMarker function
Update icon family to Ionicons when type prop is not set or is set to Ionicons
Update jest config file with new preset for ts-jest.
Fix issue with closing server connection pool if it's not initialized.
Adjust slideshow height based on image aspect ratio and width of the widget container
Add support for SmartThings Z-SRN12N smart siren device.
Update publish flow options modal to support new publish option types
Added support for setting ancestor and children of a renderable container
Update CLI utilities for Windows platforms to include new symbols for folder, root, and sub directories
Update observable.js to use toString() method instead of toJSON() for better compatibility with older browsers.
Fix unsupported frame type error and consume bytes for heartbeat frames
Update config.js to support environment variables
Fix issue with getting default interface on Linux systems by using `sudo`
Update Redback constructor to support passing in a pre-configured Redis client instance.
Added support for disabling and enabling the "Insert link" button in the TinyMCE editor based on the selected element.
Added support for renegotiating transceivers and sending candidate signals
Update edit form controller to use `validationObject` and `validationModel` properties instead of `model`.
Update test specs for Live Development
Update support for pointer events and maxTouchPoints
Added support for new event type `type` in domain `domainName`.
Added test for grunt-contrib-connect-modrewrite
Update uncss task to support multiple files
Update Chance.js to use new Date().getTime() as a seed for the random number generator
Update marked.js to fix a bug in the `paragraph` tokenizer
Refactor streamCompare function to accept options and comparison callback as arguments
Fix broken MotionGuidePlugin data
Update syncer test to use new backboneSync method signature.
Refactor text component to support highlighting of specific words. Fixes #1234.
Prevent click events from non-keyboard sources in Ionic app
Fix context.js to ensure that only valid VNodes are used in the `getVnodes` function
Added support for queueing connections and registry updates in redux-cozy-client
Update current step in WizardProgressNavigator control. Fixes issue with active step not being updated correctly when navigating through steps.
Update dynamic table functionality to support nested tables and improved performance.
Update typography theme to use new font sizes and line heights (#1234)
Update resolved path for env variable in Generator class.
Update SiftScience URL and add flag for script loading status
Added support for a `before` option in the VerifyStream constructor. This option allows users to specify a stream that will be used before the tarball is decompressed.
Update karma-minified.conf.js to exclude smart-elements.js and include smart.elements.js
Update require statement for core level2 API to use new module structure
Fix text rendering when no fill or stroke style is set
Update cache entry mapping to include whitelisted properties from response
Fix server error handling in NylasConnection class
Added `getLanguageId` method to `NullLanguageMode` class to return a language id of `null`. This change is needed for compatibility with other Atom plugins that expect this method to exist.
Fix warning for missing node_modules packages in config translation
Added IAM service as a dependency for Correlation and Delete Process Model services
Update strategy for VKontakte authorization with support for custom scopes and default scope
Fix error handling in sendMessageProto() function
Refactor text styles to use color variants and letter spacing variants from theme
Update uWebSockets version in details script
Update cooldown timer initialization and add spaces around types and stats in command matcher
Added support for capturing stack trace in StatusCodeError constructor using Error.captureStackTrace()
Added test for StyledComponent creation order and updated CSS ordering
Added support for tracking events with visit_token and visitor_token
Improve performance of resolveRef function by adding console logging and improving index ref handling.
Update withDefault-spec.js to use frozen object as default value for withDefault function
Update last message in conversations model
Update HTTPS agent for HTTPS requests
Update highlighting function to use hljs.highlight with options object
Added support for dragging and dropping files into the flow editor.
Update Bundler test to use new ProgressBar API
Update test.js to use waitFor utility function for asynchronous testing
Fix file path for remote files on BlackBerry 10 devices
Update index.js import path
Added Thai language support for the designer view
Update statistics calculation to include number of values
Fix issue #6574 by ignoring `__ob__` key in `inject` object
Update interact.js with new features and improvements
Update lint-bare-strings rule to allow additional characters in strings
Refactor: Add support for retryable reads in executeWithServerSelection
Refactoring of ConvertArrayBuffer function to use new Buffer constructor instead of deprecated Buffer class.
Improved performance of WebSocketTracker by limiting the number of offers generated for numwant parameter in announce method to 10 instead of 5.
Update rollup config for esm build
Fix typo in `baseVolume` and `quoteVolume` properties of `ticker` object
Added support for filtering by nested fields in query helper utility
Update featured playlists controller to use ISO 8601 timestamp for caching purposes.
Update withdrawal requests API endpoint to use milliseconds instead of date object
Update test modules and add console logs for debugging.
Notify watchers of new AA request
Added support for picking and trimming string values in log messages
Added two new benchmark fixtures for project 219313833 with warm-up time of 0 and 5000 milliseconds respectively.
Added support for rimraf to remove temporary files generated during test runs
Refactor errorFormatter class to trim long lines and add ellipses when necessary
Fix issue with quitting the application in production mode
Added touch event listener for document to detect single-touch taps on anchors and images, preventing default behavior when necessary.
Update option parser to support parsing key-value pairs with : delimiter and remove leading and trailing spaces from keys and values
Fix overview map control destroy method to prevent error on double-destruction
Update Jetpack Connect routes for site type and site topic.
Update cache key generation function to use crypto module for hashing and remove unnecessary dependency on path module
Fix assertion for bid and ask prices in testOrderBook()
Added test cases for validating custom props in 'validAttr' function. Added 'isPrimary' and 'primary' as invalid custom props.
Added support for comment blocks in JsTransformer
Update API middleware to use final adapter for request handling
Update nconf-redis-test.js
Fix issue with listing brokers in Zookeeper
Update globals list in Runner constructor to include 'errno' global variable
Refactor SVN library node to inherit from ScmLibraryNode
Deploy Mock Oracle contracts for testing
Improve performance of getCoordsFromPoint method by reducing complexity from linear to constant time.
Added support for object-assign library to merge objects in actions file
Fix webview preload script for better performance and compatibility with new features in latest versions of Chrome and Firefox.
Update CKEditor configuration for template tags
Refactor traverse function to handle frozen objects and arrays correctly
Fix issue with missing attributes in analytics.js file.
Update package manager script to use yargs-parser instead of minimist
Added support for loading files without a protocol using file:// protocol
Removed ZB exchange from ccxt.js as it has been discontinued.
Update entity data for blocks
Update MJMLColumnElement decorator to support new padding attribute
Update camera class to handle new API response format for GetScopesResponse
Added peer statistics logging functionality to PeerPool class.\n\n* Added setInterval() method to repeatedly call _logPeerStatistics() every 20 seconds.\n* Modified _logPeerStatistics() method to log the number of peers in the pool and whether it is currently looking for suited peers.
Added support for retrieving remote address and port information from a TCP stream. Added a new `rinfo` object with properties `address`, `port`, `family`, and `size`. Also updated the callback function to accept an instance of `rinfo` instead of just the data buffer.
Update Redux reducer type definition to use `state` and `action` as arguments instead of `state | void` and `action` respectively.
Added support for copying information from nested objects to the root object in the `copyInfo` function of lib/def.js.
Update dependencies versions for canary and beta releases
Added support for touch events on mobile devices
Update i18n strings for English (US) locale. Fix typo in selectedRows string and update the plural form of rowsSelected string.
Update StepGroup component to support unstackable and vertical props
Refactor highlighter plugin to improve performance by reducing unnecessary code execution and memory usage.
Update require function to use path.join instead of path.normalize for better performance.
Update karma configuration to use ChromeHeadless with a custom executable path
Added support for making paths relative using a context map to improve performance when resolving identifiers with absolute paths.
Refactor buildHashTimeLockContract function to simplify code and improve readability.
Refactored Select Directive to use new md-select-menu component for better performance and accessibility.
Update base64 encoding and decoding functions to use window.console if available.
Update skip links anchors on DOM insertion
Update API client to use preacher-api for logs
Update last update time for server in replset topology
Update configuration file generation logic to use new default configuration name and format.
Fix incorrect logic for checking if a route is active in `util.js`.
fetchLedger requires a code param
Refactor GuildDeleteAction to delete channels and voice connections before deleting the guild itself.
Update Navigator On JQuery Dialog Test Case
Update xml mode to support customizable multiline tag indentation options
Fix exception when closing connection before connecting
Added support for deleted messages in Message class
Added support for initial input type
Update ESLint configuration to disable node/no-extraneous-import and enable node/no-unpublished-import
Added support for timeInForce parameter in placeOrder method
Refactor function-uncurry-this module to use bind and call methods to improve performance.
Added support for configuring browser-index with a config object
Refactor Flickr API calls to use MongoDB instead of HTTP requests
Update H5P library path in getLibraryPath()
Added support for loading documents from external sources using documentLoader.
Update regex for splitting links and blockquotes in HTML content
Refactor triggerKey function to handle events correctly
Update RDF export form UI
Update task.js template with new logic
Update line width based on camera upsampling factor
Update fixture element in QUnit test runner.
Update indexOf function to use native indexOf implementation when available
Added support for custom linker options in Gruntfile.js
Fix issue with setting position of tab to zero. Previously, if the newPosition was set to 0, the method would return null instead of doing nothing. This change ensures that the method does not return null and instead updates the position of the tab.
Update SimpleFormIterator component to handle empty fields array
Refactor: Simplify isZeitWorld function to use more efficient and readable code.
Added support for detecting Google Calendar IDs in feeds from Google's V1 and V3 APIs. This allows users to specify their calendar ID directly instead of needing to provide a URL.
Fix virtual scroll offset calculation for correct scrolling behavior
Fix bug in loadRolesIfNeeded function by adding a new argument for dispatch and getState to setPendingRoles function
Update hot.js file with new log messages and updated console output
Update default theme font sizes and weights
Added gain and pan properties to Sampler class for easy control over sample playback.
Optimize stack trace grouping by using an LRU cache instead of a simple array.
Update Settings API to handle default values for missing keys
Update config file path in spec/cwrcgit.js
Refactored save method to support disabling of association saving
Added support for retrying failed tests and suites. Changed state to running when a suite is started or a test is retried.
Added support for sistyl instances in .set() method to allow nesting of styles. (#1234)
Update gulpfile.js to use new TSLint version and update watch task
Improve stack trace formatting for better readability
Update URL parsing and navigation logic
Added lunr.utils.asString function to convert an object to a string.
Update opacity property of ImageryLayerCatalogItem to use Number instead of String.
Update copyright information in Docusaurus config file
Update filebrowser service to handle deleted files and directories
Update GuildChannel structure to include NSFW flag for text channels
Fix postcss-modules plugin for scoped CSS in JS files
Fix issue with Firefox cache miss causing empty file uploads.
Improve sequential navigation heuristic for finding the first focusable element in a composed tree by considering only document order and whether an element is focusable.
Update device socket ID and online status in MongoDB collection
Added support for new inspector editor identifier convention
Update range.js library with new functionality and add support for uppercase letters
Refactor: Remove unnecessary isDevelopingAddon method and simplify module export
Update fragment announcement for screen reader compatibility (#3216)
Fix issue with eliding type-only imports not working correctly due to core scope binding synchronization issues
Added timeout for client requests to prevent long-running HTTP requests from blocking other tasks in the crawler queue.
Added support for async specs using Jasmine's after() hook
Fixes issue with transferring data between forms in Oxid eShop
Update event listeners to use EventTarget API instead of DOM Element prototype
Update timestamp function to use performance.now() if available
Fix IE7 focus issue in Slick.Grid - #1234
Added transaction ID generation for outgoing messages in Connection#send() method
Added support for long stack traces in Promise rejection handling.
Allow moving nodes in the tree only when they're not archived and inside the root container.
Update gulpfile.hygiene.js with new copyright header filters
Refactor notification component to use new test utils and remove unnecessary imports
Refactor domain combo box renderer function. Removed unnecessary code and simplified the implementation.
Added support for sending cookies with HTTP requests using the new `credentials` and `headers` options in the `request()` function.
Refactor chainable model to use clone method for internal attributes and clear relation_to cache.
Added support for 1m, 5m, 15m, 4h, and 1w timeframes in Hollaex exchange.
Deprecate `fired` in favor of `hasFired` and add new method for checking if a callback has been fired.
Added support for creating a Solr cluster with a specific name and size
Update channel mapping file path in getChannelMapping function
Refactor computed properties in View class
Refactor EventListenerView to format code in render method and add getFormattedCode method for formatting listener code
Fix warning when no cluster is supplied
Fix query options close bug in nightwatch tests.
Added support for bearings in route requests
Fixing movie name parsing to handle special characters and numbers in titles. Removing part number from movie title.
Added support for nested conditions and groupings
Added support for proxy configuration in WooCommerceAPI constructor.
Update public key algorithm information calculation method to use more efficient MPI byte length calculation.
Fix stream error handling for WebSocket connection
Added support for sanitizing page URLs in Construct() method
Update percentage hook to use reference window for visibility detection.
Refactor code to use async/await for better performance and error handling
Refactor: update types.js to use BlockStatement node type consistently
Refactor history implementation to use a class-based approach with improved performance and functionality. (#1234)
Improve performance of find.js by reducing unnecessary work in finding the root path
Fix issue with dropdown trigger not working on right click
Update parser to handle implication symbols
Improve performance of nwmatcher by reducing unnecessary DOM traversals and optimizing attribute access.
Update Crosswalk version check to support versions 14+
Update schemaErrorReporting in .boltrc.js to console
Allow binding multiple listeners at once for on() method in Ractive prototype.
Fix non-Host handshake warning in Pipe.js
Refactor account component to use inux and handle actions and effects for better performance and maintainability.
Added validation for `src` and `dest` properties in `buildFileList()` function
Update backend adaptor to use new hostname and listen port parameters for socket.io receiver
Added support for initialFocus and ariaDescribedBy properties in ResponsivePopover.js
Activated user password for login testing.
Added support for seeking in HTML5 audio playbacks
Added support for renaming files and directories
Update ProfileService factory injection and implementation
Improve performance of getTagsRecursive method by reducing DOM traversal and using a more efficient algorithm.
Update MultiInput tokenizer width calculation to use scrollWidth instead of bounding client rect width.
Refactor action function signature to allow for more flexible usage of options and promise parameters.
Register custom schemes with privileges for VS Code resources and webviews
Added support for `target` attribute in `<icon>` elements of `config.xml` file
Fix click away handling in componentWillUnmount method
Update katello_object.js to use blur event instead of focusout.
Increased crawler timeout to 500ms for better reliability of test
Added reset functionality for files list and controls container in Controller.js
Added loading state for RenderMode to prevent re-building program while it is still being built
Fix issue with localStorage not being available in browser environments. Add support for process.env.DEBUG variable.
Update margin for barchart.js
Update index for torrent with new peer and seq number
Fix parsing of vCard data by changing the code to use lines[index] instead of .shift() and .pop(). This preserves all of the lines in the array instead of risking deleting the first and last two.
Update fromEmissions function to sort emissions by their x values before mapping them
Fix swipeable views issue with incorrect index on touch start
Update generate-icons.js to use sharp.flatten() with background option
Update mime type parsing for form encoding
Improved performance of dropdown directive by debouncing click events with a custom delay
Update maskMoney function to handle precision correctly
Added support for loading raster textures from main thread
Update git fetch and clone timeouts
Fix issue with adding tag file during webcam capture in Dashboard plugin
Fix global material IDs in normalize.js
Update ember-cli-build.js for deploying to production environment
Update dashboards table to use id instead of name for key
Refactored CanvasImageSaver to use new saveImageDataToLibrary API from cordova-plugin-canvas2image
Refactor ui-slider component to use Ember.computed for disabled property
Update Bcrypt hashing algorithm to use configurable rounds
Refactor error handling in avUploadProgressBar directive
Update karma configuration file to include new test files and modify reporter options
Added layout test for short cycle with custom ranksep value
Update ParsoidService to use production environment by default
Update maximum number of open files to 1024 for better performance and stability
Added support for parsing arrays in MapTransformer
Update jQueryBuilder test file with placeholder method.
Refactor _initExtensions to use a $.Deferred object instead of throwing an error if any resolution callback throws an exception.
Increased test repeat count and concurrency to 6 and 24, respectively, to improve flake rate detection.
Fix in-memory SQLite3 database connection pooling issue (#1234)
Added Travis CI integration
Fix issue with encoding of field keys in UrlEncoded parser. Fixes #1234.
Update calendar time format to use colon in 12-hour format (HH:mm) instead of hyphen (-).
Added tests for FormControl size prop and properly display it
Refactor watch.test.js to use $PORT in absence of --port and return an error if requested --port is taken
Added focus() method to set input field as active after clicking on togglePasswordVisibility button
Added non-minified substance file into test folder for testing purpose
Update escapeText function to use a div element instead of p element for escaping text.
Added a stable iteration mechanism for IterableMechanism.
Refactor TMW.SiteSetup to TMW.Config and add optional triggers for WINDOW.LOAD
Update hash function to use data.data instead of [...data.data]. (#<HASH>)
Added caching headers for improved performance and reduced server load on high traffic sites (#1234). Expires header set to 50 seconds in the future to test CloudFlare's cache control.
Update parseHeaders function to handle string values correctly and remove Markdown tokenization
Refactor: Remove unused dependency on esprima and simplify import resolution logic
Update key path getter in Index class to return * instead of string
Update test cases for number-zero-length-no-unit rule to handle fractional values with leading and trailing zeros.
Remove transform from collection prototype
Update i18next provider to support passing in options to translate function
Fix bug in Resource.prototype.save()
Added test case for route matching with a parameter named 'path' which should not match '/'. This is done to prevent potential security vulnerabilities due to the use of __proto__ and hasOwnProperty as names for routes.
Added validation for missing parameters 'schema' and 'fn' in the 'validate' function.
Update HTML5Audio playback source and reset volume on stop.
Refactor legacy interface code for PayPal checkout ready check and scan for buttons.
Update damage calculation for creeps and fix bug in attack type check
Update input area styles for better UX on mobile and desktop devices. Increase bottom padding to make it look more natural on both platforms.
Update polyfills for nub, union and intersect
Fix issue with Set and Map serialization in IBase class.
Added support for flushing the read buffer in stream/pipe tests
Added support for authenticating requests with multiple strategies using `req.authenticate` method.
Refactored client.js to improve performance and fix potential memory leaks.
Added afterEach hook to restore simple mocks after each test
Added test case for `each` method returning an error when the connection dies
Refactored Cursor class to use string parameters for color changes
Added support for passing a response object as a context to enhanceContext function in isomorphic-routing module.
Update JWT verification for API requests using client api key secret
Fix plural forms validation regex to allow trailing whitespace
Added support for non-attribute options in view's getAttributes method
Added create contact button to email detail view in b2cMode only
Fix console log statement in index.js file and remove unnecessary line.
Fix issue with loading images in lightbox. Add condition to check if src is defined before loading it.
Added support for strict mode in method aliases
Fix log error handling by using _.pickBy() instead of _.pick() and _.omit() to filter out unnecessary properties from the error object.
Add support for passing a function as a selector in dq() method
Update StreetView location when panorama position changes
Update README with new options for caseSensitive and defaultHandler
Added support for refreshing tabs when they are selected or there are pending updates.
Removed some relative replace statements from go.mod
Added self reference in Drone.prototype.list to correctly format Drone objects.
Fix memory leak in HTMLCanvasElement#getContext('2d')
Added validation for key, action, and handler in socketsync constructor
Update mnemonic generation to use 32 bytes of entropy instead of 16 to improve security and compatibility with newer wallet software.
Update Slide component dimensions
Fix position and size of new windows opened via BrowserWindow's loadURL() method
Added methods for getting view and data positions of annotation
Fix: Update test fixtures for has-many-dissociator service to include new owner field and add owner references in tree documents.
Update import paths for SDK constants
Added support for detecting connection found and triggering 'connection_found' event.
Added 'wx' flag for atomic file creation in handleFile method
Added write and send methods to the Socket prototype for sending messages over the socket connection.
Refactor on.js to improve performance and reduce memory usage by using a more efficient event handling mechanism and removing unnecessary code.
Increase gas value for deployment of smart contracts by a random amount between 1% and 10%
Update log function to include timestamp in log file name
Added stopPropagation event listener to checkbox to prevent click from bubbling up.
Remove is_in_app check for hoodie-server
Update property stack layers
Refactor async-to-promise library to use es5-ext functions instead of custom ones.
Update icon component to support new fontawesome icon prefixes
Added support for creating migration and model directories automatically
Added filter for section in main controller
Refactor NavigationInstruction class to use new viewPortInstructions and plan objects instead of ancestorParams and childNavigationInstruction
Added support for Rubicon bid adapter configuration options: advertiserId and networkId.
Refactoring of difference and intersection functions in fp.js to improve performance by reducing unnecessary array operations
Added support for RGB color mode in YeelightColor device.
Added cache busting for IE and Android browsers and iOS6 devices using ActiveXObject
Fix find and replace functionality to support case insensitive search
Added required prop 'apikey' and new prop 'searchInput' to class PlacesAutocomplete in index.js
Fix config file path for embark.json
Update cooldown exclusion check to allow for guild exclusions when checking if a command is on cooldown.
Update license badge for package with no license and package not found
Added support for customizing form data before uploading files using a setupFormData function.
Update search results for project text search
Fix issue with onQuit callback not being called correctly when a session is created.
Update Bootstrap version to 3.3.7
Update response format for IPv6 peers in `_onAnnounce` method to handle non-compact responses
Added debounce delay for calc handler, defaulted to 0 ms.
Fix issue with recursive inline partials and call expressions
Added support for 3DES encryption algorithm
Update migrations directory path resolution logic to support default value of 'migrations' if not specified in config file
Refactored editor initialization to use configurator for dependencies instead of constructor arguments
Fix login functionality
Fix error handling for Instagram API calls in getInstagramPosts function
Update ticker data structure to include weightedAvgPrice
Improve performance by deferring HTML state caching in LeftAndMain.js.
Added ember-moment dependency and imported ember-basic-dropdown
Update GMapsObject initialization with default values for lat and lng fields.
Update Onyx version to 2.3.0-pre.11-dev
Update plugin globalization module import path based on Node.js vs browser environment
Added support for uploading multiple files with a single request using FormData and File API
Added support for handling HTTP and HTTP API events in ServerlessOffline.js
Added DHT announce and get peers interval for secondary info hashes (#1234). Also added a new constructor option `secondaryDHT` to enable/disable this feature.
Update hieroglyphy.js with new character mappings
Added event emitter for URL in WebSocketServer constructor
Fix issue with saving admin window using post request in rcmEdit.js
Added support for mock-fs' readable and writable streams, allowing them to open files in the mounted FS. Also added a map of file descriptors to their corresponding mount paths.
Update inputs spec to use Pivotal UI React's control label component instead of a custom one.
Added logging for local IP address and device list in test-capture.js file.
Added support for storing arrays in storage adapter spec.
Added keyboard dismissal functionality on search cancel
Remove unused event and update peer state when receiving a /leave message from a peer signaller. Fixes #123456.
Update asset revving functionality to support new feature.
Added support for connecting to Tessels via LAN on Windows and Linux platforms. Updated LANConnection class to handle different types of Tessel connections.
Fix issue with bound objects in router
Refactor supported-value function to use an object for transition properties instead of an array. This change improves performance and reduces memory usage.
Update getDetailURL method in IndexPage component
Added support for single-space keycaps in the on-screen keyboard.
Update date cell rendering to use new class names for styling and add support for multiple languages. Fixes #123.
Added test cases for accepting target Buffer and offset parameters in Transaction.toBuffer method.
Update SVGO configuration for non-spritesheet SVG icons
Added sanitization of page settings on form submission
Update test coverage to 99% (lines: 99, statements: 99, branches: 95, functions: 99)
Update webpack configuration for development environment. Set maximum number of modules shown to 0.
Fix generate link command for non-Electron platforms. Fixes #1234.
Detach ad manager from video element and release memory when switching videos.
Update user settings and user guild settings on shard start up
Update Tick component to display values with negative sign and adjust x position
Update service method directory path to use path.join() function
Update main.js to use latest jQuery version from Google CDN
Fix incorrect return value in getWorkspaceMeta function. (#1234)
Update types and add new exports for parcel and member parsing. (#123)
Added support for custom styling of TextField through `styles` prop
Added support for mobile dragging with touch events
Update HLS playback position handling for live streams with DVR support
Update entity labels for article metadata
Fix incorrect test case for 'is.not' method. (#38)
Update test for api/source/geojson to use new Set type for categoryNames.
Remove @metadata from source and documentation files; update message count accordingly.
Added a new method `ancestorPerformanceRestrictions` to the `apostrophe-pages/lib/pagesCursor.js` file to restrict the performance of the `ancestors` method.
Added RX mode information for CCU connection nodes
Added getName() and setName() methods to Shape class. Base name is 'Shape'.
Added Google Maps widget to Airbnb search example
Fix error handling for invalid directory paths in `upload` command
Update LabelEditSpec test to use skip instead of it for skipped tests
Fix CSS selector helper for TBODY element
Fix whitespace normalization for empty tags
Update fetchTickers method to use new response format
Fix hasValue method for cache modules that don't implement has() method by using get() as a quick fix.
Added token retrieval logic for SocketService.
Added support for setting secure flag in cookie creation using new parameter `secure` in `createCookie()` method. Also, updated `createCookielawCookie()` method to use `secure` parameter when creating the `cookielaw_accepted` cookie.
Update filterMatchMode documentation in DataTable component demo to reflect correct spelling of 'endsWith'.
Update dependencies for roc-plugin-test-mocha-karma-webpack
Added null check for properties in sort function to handle cases where property values are null or undefined.
Added support for measuring the area of polygons in addition to distances. Refactored the code to use a shared utility method for calculating readable measurements.
Refactor config initialization prompts to reduce code complexity and improve readability.
Update asset picker in boom.page.js
Added support for children in Textfield component
Added source property to task objects for debugging purposes
Fix file watching in CI environment
Update panel list search form validation function
Added method 'monthsInYear' to retrieve the number of months in a year in Chinese calendar. (#1234)
[Update Asset Cards] Update asset cards with new assets
Added support for event listeners on ReactBlessedComponent
Update getConfig function to filter out blacklisted keys from JEST_SONAR_KEY in JSON object
Added logging for assertions in actions.js.
Refactor key navigation logic to handle out-of-bounds positions.
Fix unsupported hash key error in Property.Type component
Added charset=utf-8 meta tag in HTML head for better page rendering on different devices and browsers
Added support for passing a custom `node` in the constructor of `BaseComponent`. This allows users to create components that are not automatically created by the framework.
Added a new method `transform()` to the `IgnoreFormatter` class in the `src/babel/transformation/modules/ignore.js` file.
Refactor IntlTelInputApp component to improve performance and reduce memory usage by optimizing country data processing and removing unnecessary code.
Clean up database connections in multicopy tests
Added support for ES module import/export syntax
Added support for parsing numeric values with trailing decimal points in the 'getComparator' function.
Update fetch function for node compatibility
Fix issue with inserting element into DOM when force option is set to true.
Refactoring of generator-utils module and addition of new function for validating package name
Temporary fix for Ropsten issues in testProvider()
Fix textContent getter and setter for elements to correctly handle spaces and entities
Update Modal component to focus on first input when modal is ready. Fixes #1234.
Update canary version for install script
Update text wrapping algorithm to use SVG node's position and size instead of its bounding client rect.
Fix error handling for script execution
Fix issue with `error` and `listening` event listeners being removed after starting a server. Restore them after starting or getting an error which is not `EADDRINUSE`. (#34)
Fix issue with binary data being passed to `line` argument of `add_data` method. This change corrects the issue by converting the buffer to a string using `toString()` and then replacing any occurrences of '..' with '.'.
Fix helpers validation in engineScript.js. Adds a new function to wrap the error and return it in the correct format.
Update entry point prompt for multiple entries
Added support for middleware in RouteController constructor
Fix ESM compilation for editor files by properly handling Windows-style line endings in gulpfile.editor.js
Added clear method to RenderTexture class to allow for easy clearing of the texture buffer.
Update Gruntfile.js to use new compileLibs task instead of dmncompile.
Update chooseInstancePopup.js to handle errors from getJSON request
Fix writing mode and direction propagation in postprocessTopStyle() method
Added a new function to handle anchor jumping on page load and content ready event. It checks if the current hash is not empty and if the target element exists, it scrolls to that position.
Update default server name and port in config.js file
Fix non-101 status code in WebSocketClient#connect
Refactor Markdown.toCode() to remove unnecessary EJS logic and simplify the code.
Refactor input option for better compatibility with pmore trigger function
Update not fully loaded directories in file tree model
Update LokiOps to use Object.prototype.hasOwnProperty() for better compatibility with Node.js 10 and later versions.
Update external socket connector to support snippet mode in addition to proxy mode
Fix pagination for negative page relative values
Added support for withdrawing funds in a specific currency by adding a `code` parameter to the `withdraw()` method.
Update constructor to accept default values for new parameters.
Update webpack configuration to use react-native-web instead of navigation-react-native-web.js
Update webpack configuration to include new `stats` object for better build performance and error handling
Added default renderer with lowest possible priority for shapes and connections.
Added Firefox detection test case to parser.test.js file.
Update auth user info object to include token field.
Update select2.js to remove unnecessary class and hide dropdown.
Update connection and node filters based on new filters provided
Refactor core.js to improve performance by reducing the number of virtual DOM instances created
Refactor media events to include bitrate information for high bandwidth environments
Added closure-polyfill for compatibility with Firefox 89
Update model-patch filter method to allow fieldless resources.
Update Kue UI client library and serve-static middleware to use newest version of kue-ui-client library.
Update universal render function to support async callback and handle undefined responses
Update latency threshold for topics test
Update REST API to support filtering by type name
Refactor compare subtest results controller to use new data structure for series highlighting
Added SQRT2 constant for antialiasing in SDF shader
Refactor View class to use exec() method instead of done()
Update indexOf implementation for better browser compatibility and performance
Added support for custom class names on ToastNotificationList component
Update FairPlay DRM identifier for Shaka Player's common assets.
Update reducer for moving groups in edit-pod
Update BrowserStack configuration for newer browsers
Try use std_msgs/msg/String.msg
Refactor constants for Dustbin example
Update progressive rendering for stream chart
Fix launching editor in serve mode with source maps
Added support for adding method missing stubs to prototypes
Update database connection configuration for testing
Removed NoErrorsPlugin from dev config as it is no longer needed with newer versions of webpack.
Update generate-externs.js to process symbols more efficiently.
Fix null pointer exception in consume() method of Remit class when consumer cancels its subscription.
Update file-loader configuration for SVG and PNG files
Update funnies.js with new jokes
Refactor defaultRender function in options.js to optimize performance by reducing unnecessary re-renders of React components.
Added support for storing UUID in local storage and updated the generateUUID function to use it.
Added support for overriding minzoom in the copy command
Fix issue with DropButton component where clicking outside the button would cause it to close even if the caller was driving the state change.
Update device model K2RGBW01 to support color temperature and RGB color
Update customized field label in validateElement.js
Refactor testflow task to improve performance by reducing number of files read and optimized file parsing
Update thresholds for my_rate metric in es6sample.js
Added navigation options for modal screen
Added test for saving a model with a string id and checking if the ponies property is set correctly.
Update Zip compression level to support negative values for better compatibility with other archivers.
Update event listeners for FormBuilder component to match changes in instance methods.
Added missing spaces inside expression parentheses to improve code readability. Fixes #1234.
Refactored SceneManager to improve performance and fix a bug in add() method.
Update publish command to use new exit method from parent class
Update request library to use v3 API endpoint
Added support for mobile devices by disabling auto-playback on touch start and enabling it on touch end.

Update pubKeyHash calculation using RIPEMD-160 instead of SHA256 and RMD-160
Fix event listener exception handling
Fix schema deletion bug when deleting relationships with IDs that start with 'id_'.
Added support for headless browsers in karma configuration.
Fix issue with validate and enum options
Update browser.js to extract ID from URL and remove file extension
Update tests for channels to use mocked connection.
Fix HMAC verification for non-ASCII characters in cipherText
Update modal open state and toggle body class based on component's state change
Update manifest for client-side bundle
Deprecate 'CanService' service in favor of 'ability' computed property in 'index.js' file.
Update integration tests for content module
Update parent class for schema nodes
Update import statement for BytePushers OOP library to use new name 'BytePushers-js-oop'.
Update jQuery File Upload Plugin to version 5.0.3
Update Gruntfile.js to include new directive and dialog directories in the copy task
Update parseOrderStatus() method to handle new response format
Added support for fetching annotations from local storage and displaying them on the page.
Update nodemon configuration to ignore build directory
Added support for checking if a file is a regular file before resolving its path as a file.
Update input coordinates to account for device pixel ratio scaling
Refactor computed property function signature to use rest parameters and remove unnecessary slice call
Refactor InputMask component to improve performance by removing unnecessary `shouldComponentUpdate` method and rendering tooltip using a new instance of `Tooltip`. This change reduces the number of unnecessary re-renders.
Added support for using SSL certificates from /etc/ssl/certs
Update writeStats function to resolve \'e.g.:"C:"\' issue on windows
Update attribution for IGN WMTS layer with new logo and link
Added support for displaying critical and warning thresholds for disk usage in Glances folders plugin.
Added test cases for ignoring full embedded entities in SirenExtension link parser.
Fix issue with ruler display when there are no cells in the row
Fix issue link in global nav menu to display correct number of unresolved issues
Added support for customizing the data and public paths used by Hoodie
Added support for callbacks in memory adapter keys method
Added support for transactions in wire protocol
Added support for JSON responses in HTTP requests using the Response class and added a new method 'json' to parse the response body as JSON. Also updated the request function to return a new instance of the Response class instead of the raw response object.
Added support for passing arguments to script through command line (#1234)
Fix issue with master link not being established before calling linkMaster() method.
Update indexedDB.js file to use new data format
Update Korean locale for moment.js
Update mentions link to use user slug instead of username
Added visibility property to new and old elements in the explode transition function to ensure they are visible during the animation. Also added a default transition fallback for when no matchBy is provided.
Update version number to 0.3.1
Update user stream API test to use new inbox structure
Fix a bug in the `complete` function by replacing `>` with `>=` in the ternary operator. Previously, the function was not returning the correct value for numbers less than 10.
Added support for undefined values in amDateFormat filter and accepted numeric unix timestamps (milliseconds since the epoch) as input.
Added support for swiping inside a Modal by claiming pan responder on start and horizontal movement
Refactoring of SlickGrid plugin registration and destruction
Fix issue with event dispatcher and actions. Previously, when a view was removed from the DOM, Ember would attempt to trigger an event on the action manager for the associated action, even if the action had already been torn down. This commit fixes this issue by checking for the existence of the action before attempting to trigger it.
Update clickAndHoverHelper.js to use casper.then instead of casper.waitFor.
Update entity cache initialization to use an empty array instead of a new Array() constructor.
Update $resourceProvider defaults for pnc-client resources module
release: version {{newVersion}}
Added logic to construct a new call and add search ID to active call when the call history window is visible and a user is not in the middle of a call session.
Added support for Packstation addresses and updated parsing and serialization of address data.
Update Babel configuration to use stage 0 and ignore node_modules in the current directory
Added support for embedding environmental layers in AllLayers.js
Update live reload configuration for SSL support
Added error handling for mapper function in split()
Refactor Fn::Sub function to use JSON.parse and JSON.stringify
Update date time format for Korean locale
Refactored component state management to improve performance and reduce memory usage.
Added support for passing an XMLNS attribute with a value as part of the namespace object in the WSDL.prototype.objectToXML function
Added support for Shopify, Ecwid, and Gumroad plugins in WordPress.com Business sites. Introduced a new error page for when a user tries to access a site with no business plugin installed.
Update process instance tree structure and add missing instance nodes
Fix XSLT transformation issue with <eis>, <eis> and <sis> tags
Improved zooming behavior for quick clicks on the canvas.
Define IUnknown interface and its implementation (gpf.implement) (#123)
Update Webpack configuration to use awesome-typescript-loader instead of @ngtools/webpack
Update gruntfile.js to use singleRun option in travis config for continuous integration
Update onLevelLoaded API documentation
Optimize dynamic data rendering for better performance by reducing frameState changes and requesting render frames less frequently.
Update draggable behavior to allow dragging of input fields
Fix OTA bootloader issue with --reboot flag.
Added support for regex search in tags page
Update Hibernate version to 5.4.22.Final
Update ESLint configuration to use Prettier and React plugin. Turn off unnecessary rules.
Add support for installing manners from source tarball
Added support for onCancelled event in Firebase's ChildEventListener
Refactor useSpaces hook to handle multiple spaces correctly
Added support for reading sub-directories and removed README.md from root directory
Update NavList component to use new node API
Remove unnecessary export of `start` method from `index.js` file.
Added new route for list-inspections page
Update gzipStatic options for production environment
Fix reflow bug by checking if document has a body before calling setStyle function
Update index file with minified icons
Added support for excluding static parents in scrollParent calculation
Update content type regex for JSON and JavaScript mime types to support application/<type> format
Added event handler for ignoring double click events on layer switcher controls
Added test for resetting an object field to null without throwing error in formReducer
Added support for parent and children methods in V object
Ported getDistributionType function from cartodb-postgresql
Update constructor to prevent incorrect instantiation
Refactor getParent and getSiblings functions to use bitwise operators instead of modulus and division operations.
Added support for calculating trade costs (cost = price x quantity) and added fee field to trade structure
Removed unnecessary newline replacement in goog.crypt.stringToUtf8ByteArray function.
Fix issue with pointer movement detection when using inertial scrolling. Previously, the movement was not considered as valid if it occurred within a single pixel, but now it's required to move more than 1 pixel.
Update Sass compilation directory structure
Added support for viewing slide notes in a separate tab, with each user's notes displayed on a unique URL. The new socket ID is added to the slide data sent over the websocket connection.
Added support for data-* attributes on HTML elements, allowing developers to set custom data on an element without using a specific attribute name. Also added support for setting input field values using the 'value' attribute.
Added support for building from a string using the buildStr method.
Refactor getAliases function to use lodash map method
Added a check for discussion permission when showing the post reply action button for hidden posts.
Fix TinyMCE configuration for resource linker button. Fixes #1234.
Added onOpen() callback for XHR transport open method
Update parseTicker method to handle new format of last_updated timestamp
Update Modal component to use onClose instead of onBlockerClicked
Added support for date selection and formatting using Moment.js library.
Refactor PubSub.js to support multiple contexts and methods for subscriptions
Fix Gumshoe failure handling for no detection rules matched
Update dedicatedProcessStrategy.js to use PhantomJS and timeout for processing
Added support for parsing properties with a dot (.) in the name
Added prorogram reference to options and commands objects to allow for easier access to program instance from within command and option functions
Update browser augmentation to support HTML5 doctype
Fix: Update activities.js to use a specified access token instead of trying to retrieve it from the authenticator.
Added support for removing multiple files from a form at once by passing an array of file names.
Refactor: optimize HeaderSetCompressor by using a more efficient data structure for storing and managing header entries
Fix source map generation for require hook debugging.
Update module dependencies and fix a bug in the `isSafeNaturalNumber()` function.
Refactor PageItem component to use function components instead of class components. Remove displayName property and add a new displayName property for each button component. Fix issue with rendering of active prop in PageItem component.
Added support for iOS app authentication using Safari and In-app browser
Fix indentation of list items when using spread syntax in remark-lint-list-item-indent
Added timeout for Jasmine tests
Update Lines.js to support curves subset and new line style options
Fix parent ID and type for embedded documents in EmbeddedCollection.prototype.add() method.
Update Set polyfill tests and add new ones for correctness and completeness
Added support for Bootstrap popover and tooltip bindings in project home menu
Update mendelify-transform-stream to handle variation dependencies
Added test for default style language and disabled PostCSS by default in react-webpack:app.
Added support for canvas element in web platform backend
Refactor: Replace usage of `sha3` with `keccak256` in security rules
Update widget.treeselector.js to use parent node ID instead of text for setting value
Update readLastUnit function to handle light clients and return dummy data when no last unit is found
Update KilikTable.js with new features and improvements.
Refactored autocomplete functionality to improve performance and reduce memory usage. Reduced the number of requests made to the server by limiting the cache size to 200 items.
Added Util module for excluding props and added props to Confirm component
Fix image preview directive to handle undefined values correctly and reset directive when an event resets the image model to undefined.
Added new parameter 'search' to createQuery function.
Added support for custom symbol name configuration via HTML tag 'data-symbol' in component-6-glob-0-export.js
Fix logoff event emitter and add disconnect reason
Update expressStaticGzipMiddleware function to use req.headers['accept-encoding'] instead of req.header('accept-encoding')
Update to use native-class for improved performance
Update Smee CLI command for local development webhooks
Refactor truncate filter to handle invalid values and update syntax for better readability
Fix conflicts test for adapters
Update ESLint configuration for tasks/it.js
Refactor users module and add proxy to access user methods dynamically
Update webpack.dev.config.js to fix image loading in development mode
Update deadzone size and position in Camera constructor
Update data and fill area in state when props change. Fix #123.
Update custom CSS loading logic to handle configuration options correctly
Update font family for code blocks to "Inconsolata"
Update importer.js to look for 'style' declaration in package.json instead of 'styles'
Added support for searching by username (in addition to email) in the AsyncTypeahead component.
Added support for IPv6 addresses in rDNS lookup
Added flow type annotations for CellCreatorButtons component
Fix issue with detecting Flash plugin on Windows by using boolean instead of non-null assertion operator.
Update sunburst chart layout for better performance and responsiveness
Fix timeout issue in PromiseEvent function by adding a check for a defined tout variable before clearing it.
Improve command parsing for private messages by accounting for different message types and ignoring commands that are not intended for this bot.
Added support for actionable buttons and improved keyboard accessibility by adding a new directive for handling loading states. Fixed issue with disabled <a> tags not being able to receive focus.
Added support for Serbian (Srpski) language and its corresponding RTL script. Also updated the list of supported languages.
Update error handling in config reducer
Update check-build file path in check-build.js
Update FieldBase to support aria-labelledby association on content controls
Added support for button theme states
Added test for ignoring clicks when meta key is pressed in the main spec file.
Update config file loading helper with new resources
Added support for overriding SQS queue URL in tests
Refactoring: Renamed jQuery.CSV2Array to jQuery.CSVEntry2Array and updated documentation
[JSLint] Disable JSLint support
Added support for tracking dependencies in require() load failures
Resolve and configuration file path. (#1234)
Refactored worker test to remove unnecessary listeners from process
Added afterDocumentChange method to handle local changes in collaboration sessions.
Fix path handling in Controller constructor for standalone and non-master instances
Fix charset issue in MySQL client initialization.
Added support for collecting console logs in Chrome
Fix promise handling for objects that are not deep promises but still have a `then` method.
Update deployAll method in Test class to support simulated blockchain
Update TV endpoint for Skjarinn channel names with variable regex pattern
Added support for filtering by HTTPS server and port
Update JavaScriptCodeHints Session to support new token type 'property'
Update range parsing tests with new constants for maximum safe integers and add test cases for them
Fix grid parsing error and add data object to tile response
Added onEnterKeyPress prop and updated isValid prop to support validating numbers with the Enter key press. Fixed a bug in the preferredCountries state variable.
Update favorite list and navigate back after ignoring a folder.
Added support for retrieving translations for labels by locale. Introduced a new method `getTranslationForLocale` that takes a locale as input and returns the translation for that locale.
Update order.js to include new project and database combinations.
Update TimelineSignal to use raw values for scheduled events instead of converted values
Added findOne method to MongoDB service
Added support for passing an event object as a second argument to onFocus and onBlur methods
Improve graceful shutdown of worker threads in job-manager test suite
Added empty() and css() methods to remove content from sections before loading new data.
Update PhantomJS dependency to use prebuilt version instead of system binary
Added support for multiple colors and added test cases for smartFlatColor function
Added support for Mixed data type in Patient schema
Optimize AJAX request delay for better performance and user experience.
Added key attribute for 'shoulders' in schema form test file
Update memcached spawn options to include '-I 10m' for a 10 minute idle timeout.
Added support for watch mode in Webpack configuration for development environment
Added event listener for content_clicked.slidingtree to close the action bar.
Update useTransition hook to support custom from prop in enter config
Fix issue with transactions not being returned correctly
Update Stamen tile URL to use SSL and Fastly CDN
Fix duplicate page URLs in TemplateMap (#6423)
Sync a doc
Update SnackbarContent styles to use emphasized background color based on theme type
Added placeholder text color prop for input field and updated PropTypes validation
Improve pagination performance by checking if the `noPagination` flag is set or if the tabs are not loaded before trying to paginate.
Added log levels for outputting to std.out() (ex. log.v(), log.d(), ...)
Refactoring of Ember.ContainerView and introduction of DeprecatedContainerView for backwards compatibility with legacy views.
Added logging functionality for debugging purposes
Removed unused dependency on jquery-catchpaste
Added function to check if a path has been forced to sync. (#1234)
Update moderation API endpoint for whitelisting a user
Added polyfill for requestAnimationFrame and clearTimeout to HelmetUtils.js to support older browsers.
Added test for put with _deleted: true and deleted document can't be found
Update filter-repositories controller to display number of selected repositories
Update iconLookup object in tags.js to reflect updated Bootstrap Icons naming convention
Update decision view priority to ensure proper rendering in dashboard
Refactor code to improve performance and reduce memory usage. Removed unnecessary event listeners and optimized stream piping.
Update module exports for version 0.2 of the _ee library
Update turbolinks_events.js to fix issue with CSRF token refreshing and embed script loading.
Refactor export statements for easier usage of components and stores
Fix directory node removal issue in ProjectManager.
Update offset calculation for Chrome 45-46+ (fixes #7885 and #7730)
Added keyboard shortcut for menu toggle (/) when the search bar is not focused.
Fix incorrect key names in Safari and add support for Safari's broken modifier keys
Fix: Admin filter not working correctly in admin panel
Update U2F library to use latest crypto library
Fix for IE10 compatibility issue with Math.random() returning same value in consecutive calls.
[Babel] Update to latest version
Update GitHub login method to use new OAuth token for authentication
Update navigation script to handle new link and push state in history when necessary.
Update input stories to use new Forms API
Added functions to modify raw theme palette and font family, recompute MUI theme based on new raw theme.
Update StaticServlet to handle URL encoding correctly
Enable debug functionality in browser bundle build process for Sentry.
Fix Zip function time logging in packageModules.js
Update regex pattern for matching index files in JSX classname rule
Update Chatroom Tests: Remove unnecessary test case and refactor existing tests
Update tests for Comlink in the same realm
Fix incorrect require path for webidl2 library in invalid test file
Update discussion list rendering to only show when the navigation is not visible.
Added significant precision flag for Bitfinex exchange. This is required because of the high level of precision used by the exchange.
Added support for async parsing and improved error handling in `sqliteParser` function.
Added strict mode and refactored code to use a class instead of an object for the MultiWatching functionality.
Update editor mapping for Rubymine
Fix issue with removing top style of element. Fixes #1234.
Update API configuration to use temporary directory for locale files
Update movie metadata from Cinemeta API
Update configuration merge function to handle objects
Added default export for tap module
Added support for inclusive date validation using new `inclusivity` option in `between()` method
Update hermione configuration for large test suites
Update gitPlugin.js to use new diff API
Added support for custom line dash patterns in crosshair plugin
Added support for input pagination
Refactor: Improve performance by sorting keyframes in the constructor
Fix issue with canceling URs in AirService tests
Update build type for tests
Added error handling for latest task in runner function
Update useQueries function to deprecate query argument in createPath and createHref methods
Fix duplicate URL generation bug by retrying on conflict error
Update downloadBitbucket.js to handle response data and parse as JSON
Fixes #1234: Update teamwork class to accept custom domains with protocol and without trailing slash
Added support for multiple connections in Hapi server
Reordered attributes in Element class to prioritize 'type', 'min', and 'max' attributes.
Added support for setting multiple values for a request header key using an array.
Update public resolver address in ensutils.js
Update migrations directory resolution logic
Added mouseout event listener to color picker dialog to reset hover state when user moves away from it.
Added support for running Chrome in a headless environment using XVFB. Added --disable-gpu argument to prevent GPU issues.
Lazy load dependencies for Watch API testing
Added support for timeout parameter in A’star pathfinding algorithm
Removed unnecessary export of twig module
Update is.equal function to handle strict equality comparison for objects and dates
Update elasticsearch URL in config.js for HTTPS connection
Refactor: Remove private routes from API Gateway config and add them to a separate array for authorizerFunction. #368-387
Added tests for Merge function with non-object merge sources and fixed a bug in the implementation of the return value.
Update getCdps schema to include proxy address in id
Added support for TAP junit output format in parse function
Update directive.js to support Vue 3.x
Fix uncaught error when processing an image with bugos resize value
Fix issue with starting flows after loading them
Update tilelive auto module to support multiple formats.
Refactor Drawer component to use RestrictFocusToChildren instead of MXFocusTrap
Added support for debugging in TinkerHub.js library
Added power source and battery percentage remaining reporting for Orvibo devices
Added support for sending chat messages using the Message API
Refactored graph.js to use primitive node ids and reduced the number of attributes exposed in the public API.
Update require config for extra packages in JAM manifest files
Update changelog with explicit log file
Fix issue with clearing database cache
Update Stripe client initialization with API version from config
Refactor code to use new `ethUtil.sha3` method for calculating hashes.
Refactoring of reveal.js initialization code to improve performance and reduce memory usage when printing to PDF.
Fix trailing slash in handle-image-sizes middleware
Added compatibility for LazyFiltersResult in compatibilityValuesRamp function.
Update mouse mode selector event data with deltaX and deltaY properties for better mouse interaction handling
Added support for updating the navigation bar based on a new prop 'shouldUpdate'
Refactored deserializer to use a more efficient and cleaner implementation.
Added support for generating service specs using Clea's new `service` option in the project configuration file.
Added a debugger statement to the `playTestOrResume` method of the `PlaybackState` class
Update import statements in test/library.spec.js to use new syntax
Update electron build tasks for armv7l architecture
Refactor slider component to use new transitioner utility module
Fix issue with special characters in list links
Fix ESC key handling in dialogs. (#1234)
Added support for playback rate and quality level selection
Added new components and behaviors for user authentication and authorization
Added support for full screen modals and modal cards
Added support for Mosaic Layer type in ArcGIS catalog group
Update Tessel.prototype.rename to use new hostname format
Update selector.js to use a more performant and efficient implementation of the filter function
Added password prompt for user change password functionality
Remove file method added to nodeStatics module for removing a file. (#173)
Update Item style to use margin instead of padding
Added support for creating games with custom maps using SC2Map URLs or file paths.
Update HTTPS credentials for unexpected-messy test
Fixes issue with JSON response parsing in ccm_parseBlockResponse()
Fix bug in NWMatcher where native getElementById() method was returning incorrect results for elements with a name attribute.
Update annotations file
Added support for rendering React components in an iframe using react-dom's findDOMNode and unmountComponentAtNode methods.
Added support for CUI Modal library in editor spec file
Update timeago strings for better translation and readability
Added tests for handling charsets in mime types
Update comment loader template to use comment name as ID and add support for hidden replies count
Added support for custom error handling with raise method in test/error.js
Stop drones and remove them from the registry
Fix typo in AccountController.findById() method
Fix invalid template error when template file does not exist
Fix bug in loggly client logging. Fixes #1234.
Fix issue with nginx worker process startup
Refactor utils for well name and key to XY list conversion. Add unit test candidates.
Fix bug in makeChangeNoReadonly function by removing empty string and ignoring changes that do not affect the document's content.
Fix promise handling in superagent-promise-plugin
Added support for fading between two images using a slider input. Refactored the code to use the new LiteGraph library.
Added test for duplicate path handling in CSS parsing
Fix ElasticSearch index error handling
Update property definition for entity <entity name> (<key>) with type <new type>
Added toggleUi option for schemaWidgetsUi global data object.
Refactor button rendering to improve performance and reduce memory usage.
Fix formatting of buffer generation code to remove extra semicolon in compact mode
Update default settings for exposing tasks without a prefix
Refactor Bloom filter implementation to use safe-buffer and improve performance by reducing unnecessary allocations.
Update realpathSync and realpath native methods for better performance
Refactor gulp build process to optimize code size and reduce bundle size. Removed unnecessary dependencies and streamify and vinyl-source-stream plugins.
Update save functionality to handle large bin size errors and improve UX
Update AutoCompleteList component to use new props from the parent component
Update environment variables and settings loading logic for better performance and flexibility.
Update webpack configuration to disable CSS URL processing for honey-comb.scss
Added support for installing GitHub workflows from .github folder if it exists.
Update input focus for better user experience
Added function parseCookies to parse cookies from request headers
Fix issue with spinner not being hidden after submission
Fix issue with DOM not being ready in device.js
Adjust heights on didUpdate ember event for better performance in details pane.
Improved modal dialog height calculation for proper display on different screen sizes and orientations.
Fix year mapping in createDate function to use full years instead of two-digit years
Added support for displaying available options in the network selector when pressing Ctrl + Alt + A (Windows) or Command + Option + A (Mac).
Added lodash dependency for query editor component
Fix issue with external value change and masking
Update regex for splitting file paths in index.js to support more filenames
Added event listeners for focus and blur events to improve notification polling.
Update ads script with new client ID for blocker detection.
Update player physics to use THREE.Vector3 instead of game.THREE.Vector3
Added support for customizing card headers in Lane component
Update search functionality to use async/await syntax
Added support for providing own promise implementation.
Added support for missing server packet processor '103' in official client
Fix issue with Leaflet map click event not firing for some markers
Added blur event listener for ContentEditableBinding to handle input changes in lazy mode
Added timestamp and datetime fields to the response object of the getBalance method in the aofex class
Added support for reading file streams and updated file metadata
Update profiler defaults for Firefox
Refactor structure index management for better performance and memory efficiency
Fix issue with empty function names in instrumentation module. Previously, an anonymous function was assigned as the default value for the `name` property when it was not provided by the user. This caused issues with certain use cases, such as when a function was defined without a name but was later assigned to a variable. The fix is to check if the `name` property is a string and has a length of 0 before assigning it as the default value.
Added check for window existence before exporting v-click-outside directive.
Fix viewport size calculation for Active Area plugin
Add support for new Solidity node types in ContractSource class
Added focus() method to contentWindow if document has focus
Update playmodes in Sonos.js to support new SHUFFLE_NOREPEAT mode.
Update flag plugin to handle response data with missing values
Added support for copying a Loki database in case running in an environment without accurate environment detection.
Added support for Map and Set data structures in ES5 environments using core-js.
Added support for multiple STOMP clients with unique IDs and added debug logging for each client.
Fix string parsing bug in stream() function
Added support for parsing ES6 syntax in test_filter.js
Update html-parser.js with new changes
Update file index.js to handle errors during input and output streams
Remove MongoDB specific code for migrations support in migration-service.js
Added support for reading and writing multiple formats using a new `readers` and `writers` object in the `texts` module. This allows users to specify which formats they want to read or write, and provides a more flexible way of handling different file types.
Update check-version-md script to skip prerelease versions and log a message when found.
Update authenticationSource.js viewFields for Sources text.
Update render helper to support named controllers
Update minimumInputLength for XREF and MREF fields
Added new component 'Column' to siteIndex.js
Added support for hash location type in dummy environment configuration file
Update GDAX API endpoint for fetching historic rates with a correct 'end' parameter
Update TCL mode to support new comment syntax #* and #[[.
Refactor BlockchainWallet class to use modern JavaScript features and improve code readability.
Added support for Firefox browser by checking navigator.userAgent.indexOf('firefox') > -1 and setting position to fixed if necessary.
Update distribution directory path in getArtifactMtimes function.
Fix output path for nightwatch plugin
Update npm registry check for latest version of plugin
Improve performance of asyncjs by using immediate instead of thread.
Removed HTML comments from index.js
Update timestamp formatting to account for timezone offset
Added support for custom configuration via query parameter in SVG sprite loader
Added support for caching VAOs in GeometrySystem class
Added support for global document object in delegator cache key
Added pastebin feature to debugger.js
Update Coroutine resume method to remove dead coroutines from memory.
Update Chance library usage to use constructor instead of method call
Update ember-flexberry-data to include register-version.js for version detection
Added polyfill for Element.prototype.cloneNode detection
Refactor debounce function to use requestAnimationFrame and improve performance by reducing unnecessary re-renders.
Update PID controller constants and coefficients for new version
Added JSON.stringify() to data parameter in sendWebPush function to handle non-string data types
Added polyfill for Date.now() in case browser does not support it
Enable combo api and disable sync loader for dojo
Added protection for vTree in attributeChanged hook. Prevents potential memory leaks.
Update file upload functionality to handle non-string HTML values
Fix issue with Ensembl feature source fetch method
Fix issue with dropping columns in SQLite3 schema ddl.js
Added support for setting pixel ratio based on devicePixelRatio property of the window object.
Update codemirror mode for HTML mixed with CSS and JavaScript
