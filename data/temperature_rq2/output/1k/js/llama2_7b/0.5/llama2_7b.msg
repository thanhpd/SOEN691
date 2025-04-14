Fixing unused property in trackCore.js
Fixing bug in ConnectionManager._processNext() by removing unused variable 'task'
Added Transparent button type to begin and end buttons in ResponsivePopover
Update hand controls component
Added constants definition for Rollup plugin
Fixed issue with end date being before start date in inventory test
Added exit function for NodeJS host
Added error handling for destroying generic role
Added functionality to replace query parameters in third-party module URLs
Added template error task and updated test1 task
Resolve v-model config for improved performance
Fixed IE9 compatibility issue in dirtyforms.js
Added Windows 8 command proxy for Accelerometer
Normalize entrypoints for Webpack 4 compatibility
Added Errors module for better error handling
Update require() to use relative path
Added validationMessage cache for faster lookups
Fixed typos and improved consistency in CLI configuration
Added UUID to file uploads
Fixing bug in peity.js (#100644): Changing event handler for input elements from `change` to `input` to support modern browsers.
Fixed hackish reload of iframe in templates.js
Fixed call site tracing for asyncFunctionInitialized() by removing the trace variable and using the original call sites instead.
Update animation type for picker modal to 'fade' from 'slide'
Fixing gradient filter in index.js
Added keyDown event listener for saving selected value when Enter key is pressed
Update datatable.js to improve performance by filtering columns more efficiently (#100644).
Added data-id attribute to regions element for better tracking and identification
Added option to ShellCraft prototype
Update IconButton to use cloned SvgIcon element
Fixed bug related to delta positioning in `shiftRange` function
Minified clean.js with selectors optimizations and URL rebasing
Added multiline optimization for TextMetrics class
Added parameter 'value' to pointerMove method
Added check for package contents existence
Fixed push payload and options validation
Added new agent functionality
Added new variable NETWORK_UPDATE_FREQ and updated related code to use it
Added optional `false` parameter to `executeScript` function to support non-Path scripts
Update PlaybackState.js to export currentRunningTest property\n\nDiff:\n-----------\n@property test : Test | null; @description The current running test\n@property suiteName : string; @description The name of the current running suite\n@property projectName : string; @description The name of the current project\n+ property test : Test | null; @description The exported current running test\n
Update exec.js to use new `webpack` command
Added polling interval for Windows
Added utilities to index.js
Update GameEngine.js with physics engine optimization
Added Redis database password option to queue configuration
Added colored output option to index.js
Added configuration for bundle production
Added server and port information for fast reconnect
Fix for RangeSlider issue with _updateHandle and _fireChangeAndLiveChange calls
Added setCustomMarker() in b/directives/custom-marker.js to resolve dependency issues
Added support for Ionicons icon family in Icon.js
Updating jest configuration file to use new rootDir
Fix for issue #12345: corrected pool.destroy() call to handle force option
Added height calculation for slideshow items in content.js
Added new device with Zigbee model 'SZ-SRN12N' and supports 'warning'}__} 100755
Added confirm button text for publish flow modals
Added method for setting ancestor and pushing child element
Added symbols for Windows platforms
Fixing null value handling in observable.js
Added support for consuming bytes after reading frame
Added configuration file parsing and environment variable fetching
Fixing default interface routing issue with sudo command
Fixing Redback constructor to accept client object instead of Hash
Added functionality to disable link button in TinyMCE editor when it is not an anchor tag
Renegotiating peer connection due to new transceiver request
Added property 'validationObject' and updated dependencies for Edit Form Controller
Added require for CSSInlineEditor-test.js and updated LiveDevelopment-test.js
Added support for pointer events on touch-enabled devices
Update API to use 'chrome[eventName]' instead of 'chrome[domainName][event.name]' for event handling
Added grunt-contrib-connect-modrewrite test
Fixed bug in uncss.js by removing unnecessary blank line
Fixing bug in Chance.js by updating generator reference
Fixing bug in marked.js by adding newline at the end of paragraphs
Improve performance by removing unnecessary compare function call
Added check for validity of guide data before making decisions
Fixed test for modification of options in syncer.test.js
Added highlight style for text components
Fixed click hijacking issue in gesture.js
Update context utils to filter out null or undefined $vnode properties
Added import for connections and updated selectors
Update WizardProgressNavigator: Fix zero-based indexing for _updateCurrentStep
Updated common.js file with new dynamic table functionality
Reverted changes in index.js to use the previous font sizes and line heights for the Typography theme Irving.
Update index.js to use `__filename` instead of `__dirname` for resolved path
Update SIFTSCIENCE_URL to use new CDN URL
Added support for before option in VerifyStream
Fixed typo in file name in karma-minified.conf.js\n Original: exclude: ['source-minified/smart-elements.js'],\n Modified: exclude: ['source-minified/smart.elements.js']
Update to use absolute path for level2 module
Fixing text draw issue
Added cache whitelist check for TritonAgent
Fix for server error messages in NylasConnection class
Added language ID to NullLanguageMode
Added warning for missing dependencies in node_modules
Added IamService dependency in ioc_module.js
Added support for multiple scopes in Passport-VKontakte strategy
Fixed push message content encoding issue
Revised styles.js to remove unused color prop
Added dependency on uWebSockets.js (version number replaced with ^ for security reasons)
Added cooldownTimers array and spaces variable for better handling of types and arguments in StatsPorn module
Added captureStackTrace to StatusCodeError
Added test for respecting the order of StyledComponent creation for CSS ordering
Fixing issue with visit and visitor tokens
ResolveRef: improved index ref handling
Fixed bug in withDefault function by adding default value for missing properties
Updated messages.js to include description of last message in conversations
Added support for HTTPS protocol and default agent for HTTP requests
Fix highlight language detection
Added check for file entries in onDragOver event handler
Fixed test for ProgressBar.tick() calls length mismatch in OSS
Added waitFor() method to ensure that readySpy and rawSpy are called only once afterEach hook
Added Windows Phone platform support for file:// protocol in tests.js
Added file path to `cat` module
Added Thai language translation to the designerCtrl controller
Added length of values array to Statistics class
Resolve inject: fix for #6574
Added underscore dependency for interact function
Added new whitelist for lint-bare-strings rule
Added support for retrying reads with server selection
Update ConvertArrayBuffer to use Buffer.from() instead of manual memory allocation
Limiting the number of offers generated for WebSocketTracker to improve performance
Updating rollup configuration to use ESM output format and uglify plugin with beautification enabled
Update gateio exchange data to include baseVolume instead of quoteVolume
Added support for querying fields with underscores and dollar signs in the path
Added ISO 8601 timestamp to featured playlist query
Added new method to calculate timedelta in Bitstamp exchange class
Added console.log statement to log loaded test modules and isBuild status
Added sendJustsaying() call for AA requests in network.js
Added new method for picking plain objects (#12345)
Added two new bench fixtures to the suite
Added import for rimraf module in generate.spec.js
Added support for long lines in error messages
Added return statement in isQuit method
Added prevention of taps on anchors and images in onDocumentTouchStart
Added special case for parsing root password options in the form of old:oldRootPassword,new:newRootPassword
Added check for map div existence before destroying extent rectangle and rect events
Added support for site topic pages in Jetpack Connect
Fixed bug in createCacheKeyFunction() by removing unnecessary import of path module
Updates for test/test.js: corrected assertion order for CoinMarketCap exchange; added check for bid and ask values
Added test for custom props
Added comment block for required modules
Fixed bug in middleware.js
Fixing typo in redis test file
Fixing broken `getBrokerDetail` callback
Added globals variable to Runner constructor
Added SvnLibraryNode instance constructor
Deploying Polymath contracts on development network
Fixing line selection for huge documents by improving search complexity from linear to logarithmic.
Added object-assign() to actions.js
Fixed new line at end of file
Added protection for HTML tags and expressions in CKEditor
Fixed frozen object traversal issue
Fixing invalid event type check in analytics.js
Added support for --help flag and updated minimist dependencies
Added support for URLs without a protocol (#1234)
Updated exchanges in ccxt.js
Added key property to processor.js
Added padding-left attribute to TD component
Added support for multiple PTZ scopes
Added interval logging of peer statistics
Added a new function for handling TCP information
Updating redux types to remove unnecessary `void` type parameter
Added return statement for `copyInfo` function
Updating dependencies for canary and beta releases
Added check for existence of element in infoGrid before returning any value.
Update i18n/en-us.js with minor changes
Reverted changes to StepGroup component
Fixing bug in highlighter plugin by updating `prevSelection` variable from `var prevSelection; `to `var prevSelection = [];`.
Update index.js to use join() instead of path.normalize() for better performance
Added ChromeHeadless as a build option for Karma
Added a map for storing relative paths for identifiers in different contexts
Update utils.js with new contract and params
Fixing typo in SelectDirective component
Update base64.js with internal helpers
Added event listener for DOMNodeInserted to update anchors in SkipLinks component
Migrating logs from nodejitsu-api to preacher-api
Added lastUpdateTime calculation for ReplicaSet\n
Update solhint configuration
Update util.js to remove unnecessary conditional statement
Added code validation for fetchLedgers
Update GuildDelete action to delete channels and voice connection upon guild deletion
Revised navigation drawer code for improved performance
Added support for null multiline tag indent past tag in XML mode
Removed unnecessary call to connection.end()
Added deleted property to Message class
Added new input type 'initial'
Disabled eslintrc.js rules for node/no-extraneous-import and node/no-unpublished-import due to unsupported features in Node.js version {version}.\nRules were previously set to 'off' in a/.eslintrc.js but are now set to 'off' in b/.eslintrc.js.
Added 'timeInForce' property to bcex class
Uncurry `function.call` and `function.bind` for improved performance
Added config module to browser-index
Fixed missing variable in Flickr API query
Update library path in H5P.getLibraryPath() to use H5PIntegration.url instead of H5PIntegration.libraryUrl
Added constants and documentLoader modules
Added support for blockquotes in Utility.split() function
Fixed keyboard event handling for better compatibility
Updates to UI Form.js for RDF Export
Fixed a bug in task.js by removing unnecessary code
Upsampling line width for improved performance
Reverted fixture changes
Updating indexOf() to use window.indexOf() for better browser support
Added custom options for sails-linker in Gruntfile
Fixing typo in setPosition() method
Update SimpleFormIterator.js with new fields
Added support for Zeit world nameservers
Added detection for Google Calendar ID in gcal.js
Fixing virtual scroll offset calculation for better performance
Added new roles to pending list
Updated hot.js with 100644 Git status\nModules that could not be updated: \n⦻ module1\n⦻ module2\n...\nNo Modules Updated.
Update default theme fonts and sizes
Added gain and pan properties to Sampler class
Added LRU cache for known grouping keys
Update Settings.js to use get() function with optional second argument
Added config file to require statements
Added saveAssociations option to persist associations during instance save
Rewrote store on run all tests
Added test for accepting sistyl instances in .set()
Update dependencies and fix typos in gulpfile.js
Improve stack trace prettification (#506-16):\n\nAdded closure information to stack trace for better readability (#506).
Update core.js to use new URL constructor for setting href
Added function for converting objects to strings
Fixing opacity type in ImageryLayerCatalogItem.js
Updated copyright notice to include Effector Core team
Fixed bug in CWD handling
Update GuildChannel with new NSFW property
Added support for scoped CSS classes in JavaScript files
Fixed issue with empty file uploads in Express File Uploader
Improve heuristic for finding first focusable element
Update SocketId: Fix for device not found or token not valid
Added import for {api} from Shared/Utilities/ in InspectorEditorRegistry.js
Update range.js with new code
Added efficiency to module.exports
Fixing issue with Screen Reader announcement of fragments
Added support for eliding unnecessary import statements
Added timeout for client requests
Added new spec.after implementation for async tests
Fixed typo in oxid.js
Updating core.js with fixes and improvements
Update to use `window.performance.now()` for current timestamp
Fixing IE scroll issue when setting focus on cells
Added transaction ID generation for send method in Connection prototype
Added stack trace logging for long stack traces
Moved node inside root container from 'inside' to 'outside'
Added resource file to ignore list (resources/linux/snap/snapcraft.yaml)
Update notification spec to remove intlMock dependency
Added renderer function for domain combo
Added credentials and headers for request
Added clearInternal method for relations in Chainable model
Added timeframes for 1m, 5m, 15m, 4h, and 1w; updated logo URL
Added hasFired method to preserve pending callbacks and clear fired status
Added cluster name and size parameters to retrieve_and_rank_lifecycle
Update channel mapping file
Fixing bug in View.js by updating reference to `computedWithProps` from `computedWithProps.getValue()`
Added functionality to getFormattedCode method to display code with proper indentation
Added new test cases for Pusher constructor with no cluster supplied
Fixed bug related to query options closing unexpectedly when clicking on a document in the list of documents.
Added bearings to route parameters
Fixed special char and word replacement in movies.js
Added new input and output examples for Condition function in test/Condition.js file
Added proxy option to WooCommerceAPI
Added algorithm information for public key objects
Update index.js to close the WebSocket connection with a specific code (1008) when an error occurs
Sanitizing URLs in index.js
Fixing percentage hook to use reference window for hiding elements
Fixing sync/async engine issue in cq
Updating types.js to use block statements
Fixed navigation back function in BrowserHistory
Update find.js to use node.path instead of root.path
Added check for right mouse button click
Added implication symbol support in Parser.js
Added support for handling objects as values in the `nwmatcher.js` file
Fix for Crosswalk version compatibility
Updating boltrc.js to use console-based schema error reporting instead of CLI-based
Added support for binding multiple listeners in one go
Fixes issue with non-Host handshakes
Added handleActions and handleEffects methods to Account component
Added support for dest file path in buildFileList function
Update backend-adaptor.js with new hostname and listen port
Added ariaDescribedBy property to ResponsivePopover for better WAI-ARIA compliance
Updated login spec with new user creation and login functionality
Added seek bar to HTML5 audio player
Added rename function to index.js
Added $http dependency for ProfileService
Fixing tag detection for Trumbowyg editor
Fixing tokenizer width calculation for better layout
Added then() method to action function
Added support for custom schemes with privileges in main.js
Added 'target' property to ConfigParser.prototype
Fixing click-awayable mixin to properly unbind click-away event on componentWillUnmount
Updating katello_object.js to fix focusout event handler for label field\n\nChanges:\n\t* Fixed focusout event handler for label field to properly update label value when leaving input field
Adjusted timeout value for crawler reliability test
Added reset files list functionality in Controller.js
Added validation for RenderMode's gl program
Added check for localStorage before configuring debug mode
Added margin to barchart.js
Update seeders index for torrent {infoHash}: from {oldMaxSeeders} to {newMaxSeeders}
Update vCard.js to use lines[0] and lines[lines.length-1] instead of .shift() and .pop() for better preservation of the array.
Sort and limit emissions in fromEmissions function
Added startX property to SwipeableViews component for better swiping experience
Fix for bug in generate-icons.js
Added support for specifying mime type in form encoding (#572)
Fixing quiet timeout in avDropdown directive
Added precision handling for money mask
Update raster texture loading to avoid flickering while loading (#100644)
Improved fetch and clone performance by adding a small buffer for fetch seconds
Fixed issue with file addition
Fixing legacy global material IDs by replacing them with placeholder materials
Adding vendor file for Ember-Native Dom Event Dispatcher
Fixing table row key naming convention
Fixed save function to include file name in success callback
Added computed property for enabled state in ui-slider.js
Update Bcrypt driver to use default round of 10
Fixed error callback in avUploadProgressBar directive
Added watched flag for materialize.js
Added test for layouting a short cycle
Added production mode default for ParsoidService
Update graceful-fs.js with new MAX_OPEN value
Added isArray method to MapTransformer
Added placeholder test for please_work function in jQueryBuilder test file
Fix for issue 1029: WARNING: This event won't fire if ANY extension fails to load or throws an error during init.
Update EnvStressTest.test.js with new test repeats and concurrency limit\n\nChanges:\n\tIncreased test repeats from 4 to 6\n\tIncreased maximum concurrent execution from 16 to 24\n\tAdded a timeout of 2000 milliseconds for each test run\n\tAdjusted the increment factor from 1.5 to 1.5 according to the diff
Added check for in-memory SQLite3 pools with more than 1 connection
Added Travis CI integration
Fixing issue with truncated keys in UrlEncoded.write() method
Fixing time format in calendar.js
Added test for proper display of FormControl size
Added test for --port being taken by another process
Added focus to input element after toggle password visibility
Added copy of non-minified substance file into test folder in make.js
Invert function in Filters prototype
Added conversion to array for efficient key iteration in IterableMechanism.clear() to avoid unstable ordering and early termination of iteration.
Update script.js with closure and module
Updating index.js with resolved promise data
Added Expires header to cache response for better CloudFlare support
parseHeaders: updated emoji parsing and removed Markdown tokenization
Fixed import for esprima and added resolveBareSpecifiers and addTSExtensionToImportPath functions
Update index.js to use a new key path for indexing
Added test for zero factional value after decimal
Fixed bug in removeTransform() method
Update translations for $i18next provider
Added setter for key in core.js
Added test for route matching '/:path' not to match any path
Added validation for missing parameters in `validate.js`
Updating HTML5Audio.js to use options object for src instead of hardcoding it
Added PayPal Checkout Ready functionality
Revised damage calculation for melee attacks
Added padding to platform-input.native.js to improve layout consistency (#12345)
Updating Polyfills.js with new polyfill functions
Added support for nested sets and maps in IBase
Added flush method to read buffer in pipe test
Added support for multiple authentication strategies and failure redirect
Added new node and updated _metaNodes and _nodes objects
Added afterEach hook for stub resolution
Added test for `each` method to handle connection death
Added tests for changing foreground and background color via string parameters
Added `res` parameter to getEnhancedContext function
Update API key secret in `nJwt.verify` function
Fixing invalid plural forms in helpers.js
Updating view.js to include non-attribute options in attributes
Added create contact menu item for B2C mode
Fixed console log for better performance
Fixed issue with loading images from sources that start with '//': Now using the full URL of the image instead of relying on the browser's default behavior
Added defineProperty for length property in runtime.js
Fixed error handling in log function
Added event listener for DOMContentLoaded in dom.js
Update StreetView location handling
Update index.js with hotkey configuration changes
Added reload functionality to main-window.js
Removed some relative replace statements from go.mod
Added self reference in Drone.running() to include own instance in app list
Added canvas2dWindowHandle check for context destruction
Update to socketsync library
Added new mnemonic length to 32 characters
Adjusted right position of slide to 800 in Slide.spec.js
Revised BrowserWindow spec to match new window size and position options
Added prototype methods for getting view and data position of annotations
Added test for deleting a document and its associated has-many dissociator with delete=true
Update import statements for PayPal SDK constants
Added `connection_found` variable to indicate when a connection is established
Added wx flag to fs.open() for improved file handling
Added write() method to Socket prototype
Fixing bug in on.js by removing unnecessary finally block
Randomize gas increase probability
Added log function to append file with new Date and time
Added event.stopPropagation() to checkbox click handler for better UI behavior
Adjusted environment.js to exclude hoodie-server directory
Update PropertyStack to include detached props in layers update
Update to functionalize() in async-to-promise.js
Update QIcon.js to remove unused regular expression 'faLa' and add a new one for ion icons
Added new directories for migrations and models, and updated config file to reflect new locations
Added filter for current section in main controller
Update navigation instruction class to remove plan property and add viewPortInstructions property
Added meta property for advertiser and network id
Fixed bug in difference and intersection functions by removing duplicate values in follower
Added support for RGB color mode
Added cache busting for IE/Android/iOS6
Revised find function to remove unnecessary characters
Added required prop 'apikey' for PlacesAutocomplete component\nExisting props: apikey, rankby, radius, type\nNew prop: apikey (required)
Refactoring config file path handling to avoid hardcoded file extensions
Added guild ID check for cooldown exclusions
Added license information for VSCodium
Added functionality to set default filename for uploaded files in Chrome
Added filepath field to search results for source ID matching query
Fixing bug in WebDriver.createSession() to properly handle onQuit callback
Update Bootstrap version to 3.3.7
Update server.js to use binary representation of IPv6 peers
Added debounce delay for calc handler (#100644)
Fixed issue with recursive inline partials
Added support for 3DES encryption
Added default migrations directory assumption for missing 'migrationsDir' key in config file
Update AbstractEditor to remove unused fileClient and saveHandler properties
Fixed login functionality by removing unnecessary password validation and adding a new response handler for the login API call
Fixed error handling in getInstagramPosts function
Fixed typos in Binance API endpoint names
Update LeftAndMain.js with bug fix and minor improvements
Added import statement for Ember-Moment addon
Update GMapsObject.js with new center coordinates
Update ONYX version to 2.3.0-pre.11-dev
Fixing plugin globalization error
Updating upload.js to include file parameter in FormData
Added HTTP event handler for function definition
Added DHT Announce Interval for secondary info hash (900000)
Added false value to characters object
Added emission of 'connection' event for WebSocket clients
Fixed bug in saveAjaxAdminWindowUsingPost function by removing unnecessary parameter and updating json type
Added hook for mock-fs' open event to map mount points and remove unused fds
Revised inputs_spec.js to correctly reference label element
Added log statements for localIP and device list
Added support for arrays in storage adapter spec
Added import for Keyboard component in search/index.native.js
Update leave.js to emit 'peer:disconnected' event instead of 'peer:leave'
Added generateRailsManifest option to AssetRev constructor
Added new LAN connections for each tessel
Update to router.js to handle bound objects in routing
Added support for transition properties in supported-value.js
Reacting to changes in index.js (100644):\nAdded getDetailURL method to handle detail URL (oldURL, searchSession, searchURL, navigator, keyEventTarget) and updated availWidth property.\n
Added space keycap support
Reorganized dateCell and monthCell functions for better organization and readability
Added test for accepting target Buffer and offset parameters in fromRaw() method
Added plugin to svgo-loader for minifying non spritesheet SVGs icons (#567).
Added new page settings sanitization logic
Added 4% coverage
Added maximum number of modules option to webpack config in dev mode
Added conditional statement to only generate links when not in an Electron environment
Released ad manager and cleared cached media element
Added support for user guild settings in Shard class
Revised Tick component to move X position from -5 to -10 for better layout
 updated service spec to use path.join instead of concatenating strings
Reorganized jQuery code for improved readability and consistency
Added `fulfill()` method to `getWorkspaceMeta` function
Added import statements for Parcel, Member, Name, Repository, and Bugs types
Added property 'style' to TextField component
Added endDragDelay method for mobile drag detection
Updating HLS playback to handle DVR status and position changes
Added labels for accepted date, published date, received date, and revision requested date. Also added person labels (add-person).
Fixed bug in isNot() test
Added 'categoryNames' field in geojson.test.js
Removed @metadata from source file
Added support for ancestor performance restrictions in pagesCursor.js
Added datapoints property to ccu-connection.js
Added getName() and setName() methods to Shape class
Added Google Maps widget to search page
Fixed error message for non-existent directory
Fixing bug in LabelEditSpec test (#100644): Skipping 'edit content with Enter key' test due to unimplemented functionality
Fix tbody selector
Fixed whitespace normalization in content.js
Removed unnecessary ticker variable
Fixing bug in cache module by changing `get` to `value` and updating `then` callbacks for consistency
Added token validation for SocketService
Fix for cookie law acceptance cookie creation
Update filterMatchMode to include new options 'startsWith', 'contains', 'endsWith', 'equals', 'notEquals', 'in', 'lt', 'lte', 'gt', 'gte' and 'custom'
Update dependencies and plugins for Roc Plugin Test Mocha Karma Webpack
Added null and undefined handling for sorting
Added calculation and display of area for polygons
Added new configuration options for format and installed modules
Updating boom.page.js with asset_rid from input field
Added children prop to Textfield component
Added source property to SuperTask constructor for better modularity and code readability
Added skip test for file watching on CI
Fixed validation in panel.js
Added function to retrieve number of months in a year
Added updateSelected_() method to ShakaDemoFront class
React Blessed Component: Improved event listener handling
Update getConfig function to filter out unwanted environment variables
Added logging for assertion attributes and variables
Added check for negative position in defaultPosition function, and set it to 0 instead (#343,9).
Added check for unsupported hash keys in Property.Type
Added Content-Type header for styling
Added requirement for node parameter in BaseComponent constructor
Added `transform()` method to `IgnoreFormatter` class
Updating IntlTelInputApp with new country data
Fixing typo in 'after' hook
Added support for ES modules import of base.js
Added check for numeric values with a trailing dot (".") and corrected them to their actual value
Updating HKP function to use new fetch API
Fixed insertion of element when negated or forced
Added validation for package name
Removed temporary fix for Ropsten issues
Fixed whitespace encoding/decoding in jsdom
Fixing focus on first input in Modal component
Updating install script with canary versioning
Added extractor for getBoundingClientRect() in textwrap.js
Added check for script parsing failure
Restoring error listeners and removing unnecessary listening listeners after code change
Added binary string replacement for better JSON serialization
Fixed issue with template helpers being a plain object
Added support for multiple module names in entry.js
Added middleware function to route controller
Fixed issue with newline characters in gulpfile.editor.js
Added clear method to RenderTexture class
Updating Gruntfile.js to use new task 'compileLibs' instead of 'dmncompile' (commit 100644).
Fixed error in get_instances.php response
Update CSS styler to only copy writing-mode and direction props from child elements to root element when body is not the parent element
Added jumpToAnchor function for responsive el anchor handling
Added server name to configuration defaults
Added response headers dumping for failed handshake detection
Fixed HTML escaping in Markdown.setStatic() to prevent XSS vulnerabilities
Added support for new special keys in InputOption
Update FileTreeViewModel to handle notFullyLoaded directories
Updating Lokijs.js with improved property accessor optimization
Added mode option to socket connector for better compatibility with some proxy configurations
Adjusted skip field for pagination
Updating withdraw function to use new endpoint
Added optional parameters to Shape constructor
Added alias for navigation-react-native module
Added stats option to ExtractTextPlugin in Webpack config
Added default renderer priority
Added test for Firefox user agent
Added token field to `createAuthUserInfo` function
Fixing issue with active dropdown not being hidden properly
Update traffic graph to show more than one node
Reorganized createVirtualDOM function for improved performance
Update mediaEvents.js to remove unused browser variable
Added closure polyfill for improved compatibility
Added fieldless resource support
Update kue-ui-client dependency and remove unnecessary fs module import
Fixing universal render callback to use Promises for better compatibility
Update topics test to tolerate longer latency in pull message delivery
Fixed todo comment in rest.route
Update compare perf chart series for new project signing ceremony
Added definition of SQRT2 constant and updated fragment shader with new variable afwidth and alpha calculation
Update View.js to use exec() instead of done()
Fixed indexOf() method in core.js to use the correct type of call for Safari and IE.
Added import for classnames module
Added support for FairPlay DRM in assets.js
Update reducer for test cases
Update BrowserStack tests to use latest browser versions (Chrome 54.0, Firefox 66.0, IE 11.0, Edge 17.0).
Updated test-rosidl-message-generator.js with new package dependencies and added a new test case
Added new page titles and locations for Dustbin
Fixing step value for Scheduler.js (temporary patch)
Fixing bug in serve.js by updating line and column numbers for Editor launch
Adding method_missing stubs for subscribers
Added new Sequelize configuration for test database
Added HotModuleReplacementPlugin and removed NoErrorsPlugin for improved build performance
Update generate-externs.js to handle processed symbols
Added timestamp property to message object in consume method
Added support for .cur and .svg files in the Webpack config
Added 4 new funnies to the list: 'Updating Updater...', 'Downloading Downloader...', 'Debugging Debugger...', and 'Reading Terms and Conditions for you.'
Fixing typo in `defaultRender` function
Added storage module for UUID generation
Added log statement for successful override of minzoom flag
Fixing internal state management in DropButton.js
Added support for colorxy in devices.js
Added label for customized field in validateElement.js
Update testflow.js to remove unnecessary console.log() call and add new line
Added my_rate threshold to options object
Added navigation options for My Modal screen
Added test for saving a model with a string id and checking if a model exists
Fixing level parameter in zip.js
Added event listener for 'removeComponent' in FormBuilder
Added space inside expression parentheses for better readability
Fixed bug in SceneManager.add() and SceneManager.switchTo() methods
Added recursiveDelete() method to Publish class
Update request.js to use API v3
Fixing touch end listener to properly enable mobile auto-unlock feature
Added import for ErrorManager and translated error messages
Update RIPEMD160 hash for public key calculation
Fixed event listener exception logging
Added check for deleting relationships when a connection is detached
Added support for headless launches in karma.conf.js
Added new test cases for Document.js
Fixed protocol and domain in less.browser.js
Added mocked connection for trigger() test
Fixed HMAC calculation in index.js
Added check for previous state of open property to avoid unnecessary class toggles
Update client plugin to use new asset module identifier format
Updating index.js to return abilities instead of calling CanService
Revised content module test cases to reflect new UI changes
Update schema.js to reflect changes in the GraphQL schema\n\nDiff highlights:\n\tAdded 'parent' property to class obj for improved code organization and readability (lines 4140-4141)\n\tRemoved unnecessary 'AbstractNode' reference in parent field (line 4140)\n\nChanges applied: @@ -4140,7 +4140,7 @@ var _Schema = {...}\n\n
Update to BytePushers OOP module
Update jQuery File Upload Plugin to version 5.0.3
Added new files to the grunt build
Added new OHLCV parsing logic for bars
Added getAnnotations call with promise to handle clear button click
Added ignore pattern for nodemon task to exclude dest directory
Added check for file type in resolveAsFile function
Adjusting input.js for correct scale factor
Computed function optimization
Fixed bug in InputMask component
Added useEtcSslCerts option to configure SSL certificates for proxy
Added resolution for Windows file path issue in writeStats() function
Update IGN attributions to use latest logo
Added size check for folders in GlancesPluginFolders service
Added test for ignoring full embedded entities in SirenExtension
Fixing ruler display for empty cells
Fixed resolved and myIssues query parameters based on current user's login status
Added configuration for PouchDB mapreduce plugin
Added callback argument to keys method
Added support for incrementing statement id in transactions
Added response class and method for request
Added new arg for script execution
Added check for connectedness of master link before calling callback
Fixed bug in indexedDB.js by removing unnecessary call to `resolve()` and returning `data` directly when `key` is not a number or less than 0.
Update ko.js with improved formatting for moment.js definitions
Update mentions index.js to use slug instead of username for link href
Added visibility CSS to new and old elements after explosion
Update atom.js to version 0.3.1
Fixed bug in test for user stream API
Fixing bug in complete function to properly handle numbers less than 10
Added new tests for handling undefined values and numeric Unix timestamps as input
Fixing issue with Modal and horizontal pan responder in SwipeableViews.animated.js
Fixed plugin registration issue in Slick Grid
Added check for jQuery events on tear-down of action handlers
Updating clickAndHoverHelper.js to fix issue with clicking element
Added array for entity cache in up-entity-registry.js
Added configuration for $resource provider in resources.module.js
Release: version {{newVersion}}
Added search ID to active call in call taker reducer
Added functionality for Packstation addresses
Fixed ignoring of node modules in cli-harmony.js
Fixed bug in AllLayers.js by removing unnecessary call to `addTo()` for embedded layers
Added filter to live reload host in index.js
Added mapper function to split() method
Revised index.js to include substitutions in template
Update dateTime.js with improved date formatting and day periods for Korean language support
Update component state management
Fixing XML namespace attribute in objectToRpcXML method
Added new error message for when no sites have plugins
Added new instance of transition in processInstance page
Preview function optimization for improved performance.
Reverted changes in viewer.js due to code rearrangement
Added interface implementation for IUnknown
Added Awesome Typescript Loader and removed AotPlugin
Added singleRun option to testConfig
Fixing binding for myFunction in core.js
Added requestRenderFrame() call in on('postcompose')
Fixing draggable behavior for input, textarea and select elements
Fix for OTA bootloader delay
Added regular expression for tag page navigation
Update dependencies and versions

Added support for unpacking manners binary after installation
Added onCancelled event listener for child moved events
Fixed issue with spaces array
Added test for reading a sub-directory
Fixed navigation list rendering
Removed unused function 'start' from index.js
Added new route for list inspections
Added configuration for gzip static serving (1 year in MS)
Reflow optimization: removed unnecessary `setStyle` call for empty bodies
Added console.error for minify error handling
Added parent reference in scrollParent calculation
Fixed content type detection in Message.js
Added ignoreEvent method to LayerSwitcher control
Added test for setting an object to null without throwing an error in field actions spec.
Added parent method to V prototype
Ported histogram distribution type calculation from CartoDB PostgreSQL
Fixing bug in Funnel constructor
Fixing tile coordinates
Added new fields to bittrex.js
Fixed newline characters in goog.crypt.stringToUtf8ByteArray
Require more than one pixel movement for pointer events
Added sass-cache directory for better performance
Added socket ID to slide notes event listener
Added support for setting element values using data- attribute for inputs
Added buildStr method for string builds
Update entry.js to use bind instead of function for creating aliases
Added check for post visibility and user permissions before inserting mention
Fixed typo in TinyMCE initialization
Added onOpen() method to XHR prototype
Fixed timestamp for last_updated in coinmarketcap.js
Added handling of click events on the backdrop (deprecated in favor of onClose)
Added _changeSelectedDate function to update selected date state
PubSub: Update subscription handling for better performance
Fixed error handling in registerEvents function
Added PhantomJS execution timeout
Added support for optional properties in `jstype.js`
Added program property to options object
Fixing DOC_TYPE regular expression to include the closing tag
Fixing token stub for activities.js tests
Added check for array length in file form handling function
Added keep property to dropped entries in compressor
Reorganized code for better performance and readability
Update to use underscorejs' isSafeInteger method for safer natural number checking
React: Update Button component
Added iOS-specific logic for Safari and In-App Browser authentication
Fixing legacy `spread` prop in `list` node
Added new spec for gulp-jasmine-browser with a timeout of 10 seconds
Added listenTo for curves_subset
Reorganized code for improved performance
Added tests for Set prototype
Added require statements for ko/handler-bootstrapPopover and ko/handler-bootstrapTooltip in projectHome.js
Fixed avoidMendelify function to correctly handle cases where a dependency is not affected by Mendelification
Added test for default style language and PostCSS enablement in indexTest.js
Added canvas and rootItem variables for improved performance (commit Hash).
Update avoid-sha3.js to use keccak256 instead of deprecated sha3
Fixed bug in widget.treeselector.js by updating line 262,7 from -self.element().val($(this).text()); to +self.element().val($(this).parent().attr('id'));
Added new unit data structure and query for last unit
Update KilikTable.js to include new value for input and select elements
Fixed limitInCache option in jQuery.autocomplete
Added import statement for Util module
Fixed image preview directive reset issue
Revised jQuery.molgenis.xrefsearch function to accept an optional 'search' parameter for more flexible querying.
Added symbol registration for ComponentJS
Update logon.js: Fix disconnected event emission
Fix for Accept-Encoding header in expressStaticGzipMiddleware() function
Added support for native class detection in isArray function
Added usage info for the CLI command
Truncate filter updated to use shorter output
Resolved conflicts in test.conflicts.js
Added support for multiple file paths in it.js
Update server.js with Proxy for users object
Added publicPath option to force loading of images in Chrome and FF
Resized deadzone of camera component
Added check for fillArea change and updated state with new props data
Added support for user agent in CustomCss
Added font family variation for Consolas and other fonts
Improved importer.js to use package.json for style resolution
Added minLength prop to AsyncTypeahead component
Added check for empty rDNS host and converted IP addresses to lowercase
Added prop 'above' and removed prop 'id' in CellCreatorButtons component
Fixing Flash support for modern browsers by removing ActiveXObject and using the native `navigator.plugins` property instead.
Reorganized columns for better layout in sunburst chart
Fixed timeout in PromiseEvent function
Added support for nickname-based command recognition
Added click event to QBtn.js
Added new language 'سرائیکی' (SKR) with ID 486 and territories '034'. Updated 'territories' field for existing languages.
Added error handling for global and daemon errors in config reducer
Added check for current working directory in check-build.js
Added aria-labelledby association to FieldBase
Added button state to theme.js
Added test for ignoring clicks when meta key is pressed
Update config.js with new resource prefixes for IAM role and API Gateway.
Added configuration extension for Client options
Added function for converting CSV strings to arrays
Added public API for JSLint results
Added functionality to handle document stack for require() loads
Added support for parsing package description as Markdown (#115)
Fixing kill signal handling in worker test file
Added afterDocumentChange method to CollabSession prototype
Fixed path resolution for standalone and non-master branches
Added support for charset 'utf8'
Fix for --multi flag in browsertime plugin
Reorganized promise chain handling for improved performance
Added deployAll method for testing contract deployment with different network options
Added support for TV channel 7 in the skjar1 endpoint
Added check for HTTPS port in rules.js
Added property type check for JavaScriptCodeHints Session object.
Added new range parsing logic for maximum safe integer
Fixed grid invalidity error in MBTiles.prototype.getGrid()
Added onEnterKeyPress prop to TelephoneInput component
Added import of navigateBack from actions/router
Added new properties to Label model
Added resource 'projects/p2/databases/d3/documents/c1-/doc1' and updated geo point coordinates.
Fixing unitty issues in TimelineSignal.js
Added findOne method for MongoDB service
Updating form-input-mixin.js with bug fixes and minor improvements
Added preferred method of shutting down worker
Added empty elements to hide trees and actions section on tab selection
Update PhantomJS to prebuilt binary
Added support for true color palettes in smartFlatColor()
Added support for mixed data types in Patient schema
Fixing delay parameter in ajax requests
Added 'shoulders' key to schemaForm item in b/test/services/schema-form-test.js
Added -I option to memcached command to specify a 10-minute timeout
Added option for watch mode in grunt-webpack configuration
Added bind event for content clicked on SlidingTree
Update useTransition to handle new props
Fixed bug in bitz.js by removing unnecessary console.error() call
Update Stamen tile URL to use Fastly CDN
Resolving conflicting permalink values in TemplateMap
Added test for syncing a document between PouchDB and CouchDB using HTTP-2 protocol
Added emphasis color manipulation for SnackbarContent styles.
Added placeholderTextColor property to GiftedMessenger component
Fixing pagination logic for Tabs Controller
Added log levels for outputting to std.out() in Nxs.js
Added DeprecatedContainerView as a new class in main.js
Added import of logging module
Removed jQuery catchpaste and cookie plugins, as they are not used anymore.
Added function isForced() to lib/sync.js
Fixing endpoint URL in client.js
Added support for requestAnimationFrame in polyfill
Added test for put with _deleted: true
Updates to Filter Repositories Controller in Bastion Katello
Added new icon names for tag colors
Prioritizing decision plugin from 30 to 90
Fixed exit code on server error
Update module exports in lib/_id.js
Added flag to force loading of UV in turbolinks events
Update import statements for easier module resolution
Refactored directory rendering logic for improved performance and consistency
Fix for Chrome bug in offset() method
Added check for input fields before triggering menu toggle on '/' key press
Added Safari modifier name check for better key identification in index.js
Update admin store with new filter
Added constant for U2F version
Added random unique ID for test runner to prevent page reloading
Optimized bundle.js by removing unnecessary plugins and modules\n\nChanges:\n\n* Removed `occurrenceOrderPlugin` plugin as it is not needed\n* Changed `resolve.modulesDirectory` to `resolveLoader.root` to avoid loading node_modules in the wrong directory\n
Update user model to include connectSid in fetch method
Update navigation.mvc.js with new Link
Update Input.stories.js to use the new 'Forms' storybook
Added function to modify raw theme's zIndex
Fixed handling of URLs with percent-encoded octothorses\n\nThe previous implementation was vulnerable to code injection attacks due to the use of string manipulation functions without proper escaping. This commit fixes the issue by using a safer approach for handling percent-encoded octothores in URLs.
Removed unnecessary debug functionality in Rollup configuration
Revised log statement for zip module generation
Added regular expression for indexing JSX classes
Fixed issue with chatroom queryAll method
Fixing bug in Comlink test
Fixing invalid JavaScript in test/invalid.js\n\nThe difference between the two files is due to the removal of unnecessary imports and the update to the latest version of `expect.js`. The `process.env.JSCOV` variable is now set to the correct value, ensuring that the correct version of `webidl2` is used in the test.
Fixing navigation visibility in DiscussionPage
Added significantPrecision option to Bitfinex exchange class
Added import for SyntaxError from parser
Added strict mode and minor improvements to MultiWatching class
Added x-mine prefix to filter mappings
Fixing bug in node.test.js by changing line -282 from `nwt.one('#id1').removeStyle('top');` to `nwt.one('#id1').getStyle('top');`
Added directory for locale files
Update addon.js to include updated metadata for IMDB IDs in the toGet array
Added support for objects in reconfigure function
Added default export for TAP module
Added validation for max and min dates in date.test.js
Added buffer size for STDOUT in hermione.js
Fixed bug in gitPlugin.js by updating the Git API endpoint for computing diffs.
Added dash pattern option for line chart
Added import of stop function from utils/event.js for better performance in QPagination component
Added sort() method to Keyframes class
Fixed cancelUR() call in Air.test.js to resolve ticketing failed error
Update to use correct version of ganache.core in public exports
Fixing promise.catch() to handle errors properly
Deprecated query argument in createPath and createHref functions
Fixed issue with duplicate entry in MongoDB by retrying generate function call
Added response variable for repository details
Added tests for custom domains in teamwork
Added SSL support for Hapi server
Added support for max, min and value attributes in Element class
Update request.js to include additional headers in XHR requests
Update resolver contract address in ensutils.js
Added test for default migrations directory when no migrationsDir is specified in the config file
Added mouseout event listener to color picker dialog to restore previous hover state upon exiting
Added support for XVFB and disabled GPU acceleration when running in Docker
Added canihaz() call for fs.notify and findit modules to enable lazy loading in watch API test
Improved pathfinding algorithm with timeout support
Fixing export statement for Twig module
Added strict equality check for is.equal function
Added port string to elasticsearch configuration
Added protection for private endpoints
Added test for invalid input in Merge function
Update getCdps function to include proxy address in ID and contract name
Fixed bug in tapJunit function
Update directive.js to remove unnecessary check and update `el.textContent` and `_vt` properties
Added test for bugos resize value in processImage.js
Fixing typo in startFlows() method
Added support for loading modules with colon in their paths
Added RestrictFocusToChildren component to Drawer
Added debug function for better logging
Added device power source and save functionality
Added new line of code to include the recipient's ID in the message post request
Updating graph.js with new API
Added support for extra packages in project.js
Added support for passing a log file directly (#100644)
Fixing bug in clear function
Updating Stripe API version and secret key in services/stripe.js\nDiff:\nOld: \n...\n-    const { secretKey } = this.app.lookup ('config:stripe');\n-    this._stripe = new Stripe (secretKey);\n+    const { secretKey, apiVersion } = this.app.lookup ('config:stripe');\n+    this._stripe = new Stripe (secretKey, { apiVersion });
Update index.js to use SHA3 for code hash
Fixed issue with printing to PDF by updating the code to wait for document.readyState === 'complete' before setting up the PDF layout measurements
Added trailing slash check in image size handling middleware
Added support for lazy filters in compatibilityValuesRamp
Added deltaX and deltaY properties to mouse event object for better tracking of mouse movements
Added 'shouldUpdate' prop to NavigationBar component
Update Deserializer class to use Object.assign() instead of manual assignation
Added support for service spec file in blueprints
Added debugger statement for playTestOrResume() method
Update tinymath imports
Added support for ARMv7l architecture in electron task
Reorganized code for better performance and readability
Fixed issue with notifications not disappearing after clicking on a document
Fixing escapable property in Dialog.js
Added playbackRate property and getPlaybackRate method to TubePlayer object
Added UserDumbComponent to the App
Added properties for full-screen and has-modal-card modals
Added support for Mosaic Layers
Added MAC address to Tessel's hostname in rename() method
Fixing selector.js to use modern JavaScript syntax
Added old password field to user command
Added removeFile method for file deletion
Added marginTop to Item style
Added support for creating games with SC2Map
Fixed key value in unexpected-messy test to satisfy Buffer([0x08])
Fixed HTML injection vulnerability by removing unnecessary HTML tags from block response.
Fixing bug in NWMatcher.js
Improving browser LiveReload performance by only writing annotations file when there are changes
Added ReactDOM.findDOMNode() for proper component lifecycle
Added CUI modal functionality to editor spec test
Update to angular-timeago.js file\n\nRevised lines: 151, 10 (added closing bracket)\nChanges: removed seconds and minute directives, updated minutes and hours directives with new syntax
Added test for handling charsets correctly
Reorganized code for better readability
Added custom 500 error handling
Added a new function to stop drones and save their process in parallel
Update account controller to handle password change
Resolve template file resolution issue
Fixing bug in Bunyan2Loggly.checkBuffer() to use log() correctly
Added check for starting worker process before logging error messages
Added utility functions for converting well keys to XY coordinates
Added check for empty text changes and skipped them if pos is equal to the current position
Fixing error handling in superagent-promise-plugin
Fixing unnecessary output node creation
Added test for duplicate paths in CSS imports
Fixed Elasticsearch index error handling in BulkWriter
Added property type validation for entity ${entityName}
Added toggleUi property to index.js

Fixing compact buffer formatting issue
Added exposeWithPrefix option to configuration
Fixed buffer type for multiCheck() method
Added native implementation of realpathSync and realpath methods in bootstrap.js
Fixed license header in gulp-build.js
Added error handling for save button
Updating AutoCompleteList component with new props
Updating environment.js to remove unused code and improve performance
Added option to processCssUrls in Webpack mix configuration
Added installWorkflow for 'push-renovate-typedoc' in postinstall/install-github-workflows.js
Reverted changes in App.js
Added function for parsing cookies in API
Fixing showSpinnerTimeout to avoid null reference error
Fixing issue with document.body not being available in non-Node environments
Adjusted heights in sl-grid.js to account for updated Ember events
Added max-height style for modal dialog
Fixing year remap issue in createDate function
Added alt key check for toggle network selector
Added import for _ and React
Update VTextField to fix issue with external value changes not updating the lazy value\n\nChanges:\n\tRemoved unnecessary `if` statement that was causing issues in some cases\n\tMoved `this.internalChange = false` statement to be executed only when the value is changed externally\n\tAdded `else if` statement to handle the case where the value is changed internally and the mask is enabled\n\tModified `masked` variable name to avoid confusion with the internal value\n\tFixed typos and inconsistencies in the code
Fixing path splitting RegExp for Windows compatibility
Added timer for polling alerts (#100644) {fetch() every POLLING_INTERVAL milliseconds after page focus/blur}
Updated right.js template with Google AdSense script
Update game.js to use THREE instead of Math for vector operations
Added rightHeader prop to Lane component
Fixed search query for Node.js on Google Cloud
Added support for providing own promise implementation
Added processor for packet ID 103
Patch for LeafletMap: fixed issue with layer movement on click
Added blur event listener to ContentEditableBinding node
Added timestamp and datetime properties to aofex.js
Added length and md5 properties to exports.readStream function in mongo.js\n\nDiff details:\nindex <HASH>..<HASH> 100644\n--- a/src/mongo.js\n+++ b/src/mongo.js\n@@ -77,10 +78,12 @@ exports.readStream = function (fileId, options) {\n             },\n             var stream = gsObject.stream();\n             resolve({...}, {length: gsObject.length, md5: gsObject.internalMd5});\n         });\n     });
Update geckoProfilerDefaults.js with new buffer size
Update indices.js to remove unnecessary properties
Added check for empty function names
Added isTouch variable for better handling of touch events
Added fallback for viewport size when none is provided
Added support for ImportDirective and StructDefinition in contract source
Added focus event listener to content window
Added \'SHUFFLE_NOREPEAT\' play mode to Sonos.prototype.getPlayMode()",\n
Fixed bug in `extract` function by adding a conditional check for `item.data.activity` before mapping it to an array.
Added support for environment detection in Loki.prototype.copy() for accurate database cloning
Updating core-js dependencies for better compatibility and performance
Added a new ID variable for the Stomp client
Fixing string escape issue in index.js
Added support for CLI-Color in test_filter.js
Added elenpi.htmlParser variable and updated module.exports
Added error handling for input and output streams in index.js
Added support for migrations in other databases
Added module.exports.readers and module.exports.writers properties
Added check for prerelease versions in CHANGELOG.en-US.md
Added warning for sponsors about selecting sources (#100644)\n\nThe `text` field in the `sources` property of the `Configuration/_config/authenticationSource.js` file has been updated to include a warning for sponsors about selecting sources. The new text reads: 'Sources that will be associated with this source (For the Sponsor). Only sources able to find an email address can be selected.'
Added support for controller lookups by name
Revised jQuery.molgenis.xrefmrefsearch.js to support minimum input length of 1 for XREF and MREF fields.
Added new component: Column\n\n(Diff details below).
Added ENV.locationType and updated CSP to include new sources
Adjusted limit in getHistoricRates to 300 (-1 instead of 413)
Added support for commenting out code with # comments in TCL mode
Added functionality for second password in BlockchainWallet class
Added support for Firefox on iOS
Update get-artifact-mtimes to use process.cwd() for distribution directory
Fixing typo in nightwatchPlugin() - changing 'env' from 'default' to 'defaul'
Update to latest version from npm registry
Fixing asynchronous callback execution for improved performance
Fixed <!-- --> comments in index.js
Fixing timestamp column value offset for timezone correction
Added configuration option for SVGSpriteLoader
Added support for returning the vertex array object from the `activateVao()` method
Added global document variable for DOM delegation cache
Added pastebin feature for debugging
Added _remove() method to Coroutine prototype for proper termination handling
Added support for seeding the random number generator with an optional `seed` parameter
Added import of 'vendor/ember-flexberry-data/register-version.js' in included function for Ember-Flexberry Data module
Added check for 'checked' property in cloneNode
Debounce function improvements
Removed unnecessary variables and calculations
Added JSON.stringify() for data in sendWebPush() to make it a string before sending notifications
Fixing Date.now() support for older browsers
Fixed bug in combo api handling cache bust query args
Added protection for VTree in patch.js
Fixing HTML sanitization issue in drop.js
Fixed error handling in EnsemblJSON.fetch() due to changed status code threshold
Added column wrapping in SQLite3_DDL.prototype.dropColumn
Added pixel ratio setting and updated canvas creation for better performance
Added HTML state to mode definition
